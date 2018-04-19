from chatbottest.data.movie_data import MovieData
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import os
from keras.models import load_model

class KerasBot:

    def __init__(self,questions,answers):
        self.eng_sentences = questions
        self.fra_sentences = answers
        self.eng_chars = set()
        self.fra_chars = set()
        self.nb_samples = 2000
        # dictionary to index each english character - key is index and value is english character
        self.eng_index_to_char_dict = {}
        # dictionary to get english character given its index - key is english character and value is index
        self.eng_char_to_index_dict = {}

        # dictionary to index each french character - key is index and value is french character
        self.fra_index_to_char_dict = {}
        # dictionary to get french character given its index - key is french character and value is index
        self.fra_char_to_index_dict = {}

        self.max_len_eng_sent = 0
        self.max_len_fra_sent = 0

        self.encoder_model_inf=None
        self.decoder_model_inf=None

    def extract_chars(self):
        # Process english and french sentences
        for line in range(self.nb_samples):

            eng_line = str(self.eng_sentences[line])

            fra_line = str(self.fra_sentences[line])

            for ch in eng_line:
                if (ch not in self.eng_chars):
                    self.eng_chars.add(ch)

            for ch in fra_line:
                if (ch not in self.fra_chars):
                    self.fra_chars.add(ch)
            self.fra_chars.add('\t')

        self.fra_chars = sorted(list(self.fra_chars))
        self.eng_chars = sorted(list(self.eng_chars))

        for k, v in enumerate(self.eng_chars):
            self.eng_index_to_char_dict[k] = v
            self.eng_char_to_index_dict[v] = k

        for k, v in enumerate(self.fra_chars):
            self.fra_index_to_char_dict[k] = v
            self.fra_char_to_index_dict[v] = k

        self.max_len_eng_sent = max([len(line) for line in self.eng_sentences])
        self.max_len_fra_sent = max([len(line) for line in self.fra_sentences])

    def train(self):
        self.extract_chars()

        tokenized_eng_sentences = np.zeros(shape=(self.nb_samples, self.max_len_eng_sent, len(self.eng_chars)), dtype='float32')
        tokenized_fra_sentences = np.zeros(shape=(self.nb_samples, self.max_len_fra_sent, len(self.fra_chars)), dtype='float32')
        target_data = np.zeros((self.nb_samples, self.max_len_fra_sent, len(self.fra_chars)), dtype='float32')

        # Vectorize the english and french sentences

        for i in range(self.nb_samples):
            for k, ch in enumerate(self.eng_sentences[i]):
                tokenized_eng_sentences[i, k, self.eng_char_to_index_dict[ch]] = 1

            for k, ch in enumerate(self.fra_sentences[i]):
                tokenized_fra_sentences[i, k, self.fra_char_to_index_dict[ch]] = 1

                # decoder_target_data will be ahead by one timestep and will not include the start character.
                if k > 0:
                    target_data[i, k - 1, self.fra_char_to_index_dict[ch]] = 1

        # Encoder model

        encoder_input = Input(shape=(None, len(self.eng_chars)))
        encoder_LSTM = LSTM(256, return_state=True)
        encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_input)
        encoder_states = [encoder_h, encoder_c]

        # Decoder model

        decoder_input = Input(shape=(None, len(self.fra_chars)))
        decoder_LSTM = LSTM(256, return_sequences=True, return_state=True)
        decoder_out, _, _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
        decoder_dense = Dense(len(self.fra_chars), activation='softmax')
        decoder_out = decoder_dense(decoder_out)

        model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])

        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.fit(x=[tokenized_eng_sentences, tokenized_fra_sentences],
                  y=target_data,
                  batch_size=64,
                  epochs=50,
                  validation_split=0.2)

        # Inference models for testing

        # Encoder inference model
        self.encoder_model_inf = Model(encoder_input, encoder_states)

        # Decoder inference model
        decoder_state_input_h = Input(shape=(256,))
        decoder_state_input_c = Input(shape=(256,))
        decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

        decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input,
                                                         initial_state=decoder_input_states)

        decoder_states = [decoder_h, decoder_c]

        decoder_out = decoder_dense(decoder_out)

        self.decoder_model_inf = Model(inputs=[decoder_input] + decoder_input_states,
                                  outputs=[decoder_out] + decoder_states)

    def decode_seq(self,inp_seq):


        tokenized_inp_seq = np.zeros(shape=(self.nb_samples, self.max_len_eng_sent, len(self.eng_chars)), dtype='float32')

        for i in range(self.nb_samples):
            for k, ch in enumerate(inp_seq):
                tokenized_inp_seq[i, k, self.eng_char_to_index_dict[ch]] = 1


        # Initial states value is coming from the encoder
        states_val = self.encoder_model_inf.predict(tokenized_inp_seq)

        target_seq = np.zeros((1, 1, len(self.fra_chars)))
        target_seq[0, 0, self.fra_char_to_index_dict['\t']] = 1

        translated_sent = ''
        stop_condition = False

        while not stop_condition:

            decoder_out, decoder_h, decoder_c = self.decoder_model_inf.predict(x=[target_seq] + states_val)

            max_val_index = np.argmax(decoder_out[0, -1, :])
            sampled_fra_char = self.fra_index_to_char_dict[max_val_index]
            translated_sent += sampled_fra_char

            if ((sampled_fra_char == '\n') or (len(translated_sent) > self.max_len_fra_sent)):
                stop_condition = True

            target_seq = np.zeros((1, 1, len(self.fra_chars)))
            target_seq[0, 0, max_val_index] = 1

            states_val = [decoder_h, decoder_c]

        return translated_sent

    def save_model(self):
        self.encoder_model_inf.save("encorder_model.h5")
        self.decoder_model_inf.save("decorder_model.h5")

    def load_model(self):
        self.encoder_model_inf = load_model("encorder_model.h5")
        self.decoder_model_inf = load_model("decorder_model.h5")




BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

md=MovieData(BASE_DIR+"/data/raw_data")
questions, answers=md.get_question_n_answer(2000)#138135
kb=KerasBot(questions, answers)
kb.train()
kb.save_model()


kb.load_model()
ans=kb.decode_seq("No, no, it's my fault -- we didn't have a proper introduction")
print(ans)








