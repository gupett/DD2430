import numpy as np

categories = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']

class affection_context:

	def __init__(self):
		self.affect_model = self.create_affect_model()
		self.affect_categories = 5


	def create_affect_model(self):
		# skapa en model
		fh = open('./Data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r')
		lines = fh.readlines()
		fh.close()
		l = {}

		for line in lines:
			line = line.split('\t')

			if len(line) != 3: continue

			(word, category, score) = line

			score = float(score.strip())

			if word not in l: l[word] = {category: score}
			else: l[word][category] = score
		return l


	def analyse_word(self, word):
		if word in self.affect_model:
			return self.affect_model[word]
		else:
			return 0

	def affection_for_context(self, context):
		result = {'anger':0, 'sadness':0, 'fear':0, 'negative':0, 'positive':0}
		for word in context:
			affection = self.analyse_word(word)
			if affection != 0:
				for key in result:
					value = affection[key]
					if value == 1:
						result[key] += 1

		return result

	def binary_affection_vector_for_context(self, context):
		result = self.affection_for_context(context)
		vector = []
		for key in result:
			if result[key] != 0:
				result[key] = 1
				vector.append(1)
			else:
				vector.append(0)

		return np.array(vector)


if __name__ == '__main__':
	ac = affection_context()

	print(ac.affection_for_context(['hapy', 'godt', 'bd']))
	print(ac.binary_affection_vector_for_context(['happy', 'good', 'bad']))
	print(ac.binary_affection_vector_for_context(['hapy', 'godt', 'bd']))


'''
# testa modellen
result = {}

for category in categories:
	result[category] = 0

word_count = 0

for word in text.split(' '):
	print(word)
	word_analyse_result = analyse_word(word)
	print(word_analyse_result)
	for category in word_analyse_result:
		result[category] += word_analyse_result[category]

	word_count += 1


# normalisera svaret i testet
for category in categories:
	result[category] /= word_count


# skriv ut svaret så att gnuplot kan göra ett stapeldiagram
for category in categories:
	print('"' + category + '"\t' + str(result[category]))
'''
