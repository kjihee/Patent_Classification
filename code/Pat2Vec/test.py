import glove

import glove

cooccur = {
0: {
0: 1.0,
2: 3.5
},
1: {
2: 0.5
},
2: {
0: 3.5,
1: 0.5,
2: 1.2
}
}

model = glove.Glove(cooccur, vocab_size=3, d=50, alpha=0.75, x_max=100.0)

for epoch in range(25):
    err = model.train(batch_size=200, workers=9)
    print("epoch %d, error %.3f" % (epoch, err), flush=True)