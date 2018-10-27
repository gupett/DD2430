import sys, json, math

sliding_window_size = 20
batch_size = 50

sentences = sys.stdin.read()
sentences = sentences.lower()
sentences = sentences.replace('\n', ' ')
words = [w for w in sentences.split(' ') if len(w) > 0]

i = 0
batch = []
unique_words = set()

fh = open('batches.json', 'w')

for w in words:
	i += 1
	
	if w not in unique_words: unique_words.add(w)
	
	if i <= sliding_window_size: continue
	
	window = [words[k] for k in range(i - sliding_window_size - 1, i - 1)]
	
	target = w
	
	batch.append({
		'window': window,
		'target': target
	})
	
	if len(batch) == batch_size:
		fh.write(json.dumps(batch))
		fh.write('\n')
		
		batch = []

fh.close()

fh = open('unique_words.json', 'w')
fh.write(json.dumps(list(unique_words)))
fh.close()
