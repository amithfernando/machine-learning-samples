
class MovieData:

    def __init__(self,file_path="raw_data"):
        self.file_path=file_path

    '''
        1. Read from 'movie-lines.txt'
        2. Create a dictionary with ( key = line_id, value = text )
    '''
    def get_id2line(self):
        lines=open(self.file_path+"/movie_lines.txt", encoding='utf-8', errors='ignore').read().split('\n')
        id2line = {}
        for line in lines:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]
        return id2line

    '''
        1. Read from 'movie_conversations.txt'
        2. Create a list of [list of line_id's]
    '''
    def get_conversations(self):
        conv_lines = open(self.file_path+"/movie_conversations.txt", encoding='utf-8', errors='ignore').read().split('\n')
        convs = [ ]
        for line in conv_lines[:-1]:
            _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
            convs.append(_line.split(','))
        return convs

    '''
        Get lists of all conversations as Questions and Answers
        1. [questions]
        2. [answers]
    '''
    def gather_dataset(self,convs, id2line):
        questions = []; answers = []

        for conv in convs:
            if len(conv) %2 != 0:
                conv = conv[:-1]
            for i in range(len(conv)):
                if i%2 == 0:
                    questions.append(id2line[conv[i]])
                else:
                    answers.append(id2line[conv[i]])

        return questions, answers

    def get_question_n_answer(self,max_size=-1):
        id2line = self.get_id2line()
        convs = self.get_conversations()
        questions, answers = self.gather_dataset(convs, id2line)
        if max_size==-1:
            return questions, answers
        else:
            return questions[:max_size], answers[:max_size]


