Heart-Anomalies
===============
Write up:
Which is more important in this application: accuracy on abnormal instances or accuracy on normal instances? Which dataset seems to give the best results?

I think the more important number is the accuracy on abnormal cases. I imagine a program like this can be sed as a first line diagnosis system, so the point is to prioritize as many correct case as possible. There are going to be second, third, etc lines of defense to filter out the wrong case. Personally, I much prefer having a normal heart and being check multiple time vs having an abnormal heart and stay on a low priority list.

The most accurate result seems to be on the itg data set *with default m = 0.5*. This accuracy however is heavily based on the choice of the m constant. For example:

m = 0.5:
itg 167/187(0.89) 13/15(0.87) 154/172(0.90)
orig 142/187(0.76) 10/15(0.67) 132/172(0.77)
resplit 74/90(0.82) 16/19(0.84) 58/71(0.82)
default case.

m = 0.25: 
itg 172/187(0.92) 10/15(0.67) 162/172(0.94)
orig 142/187(0.76) 10/15(0.67) 132/172(0.77)
resplit 74/90(0.82) 16/19(0.84) 58/71(0.82)
minimal impact on orig and resplit, itg improved 3 pts overall but doing worst on the abnormal case.

m = 0.05 
itg 174/187(0.93) 3/15(0.20) 171/172(0.99)
orig 144/187(0.77) 10/15(0.67) 134/172(0.78)
resplit 75/90(0.83) 16/19(0.84) 59/71(0.83)
minor improved on orig and resplit case, itg improved 3 pts overal but absolutely broken on the abnormal case with near perfect result on normal case.

m = 10 * double.Epsilon (the smallest number representable using IEEE double precision floating point data type - used 10 times to account for taking log of it)
itg 40/187(0.21) 15/15(1.00) 25/172(0.15)
orig 145/187(0.78) 10/15(0.67) 135/172(0.78)
resplit 75/90(0.83) 16/19(0.84) 59/71(0.83)
again, fairly stable result on orig and resplit, completely broke itg data set but most noticably 100% accuracy on the abnormal case

There is probably some complex statitics/math explanation behind why itg abnormal case goes down and then jumps up as m decrease. But clearly m has some effect on the final result and thus should be choosen carefuly to avoid overfitting / underfitting.

Program run really fast, which is to be expected for O(n) training time complexity and O(1) time testing per individual case.

I'm also curious on why the training set is smaller than the test set, so I flipped them around, see if it can do better:
m = 0.5:
itg 69/80(0.86) 34/40(0.85) 35/40(0.88)
orig 61/80(0.76) 30/40(0.75) 31/40(0.78)
resplit 133/177(0.75) 26/36(0.72) 107/141(0.76)
default case: overall, fairly similar result.

m = 0.25
itg 64/80(0.80) 28/40(0.70) 36/40(0.90)
orig 63/80(0.79) 30/40(0.75) 33/40(0.83)
resplit 137/177(0.77) 25/36(0.69) 112/141(0.79)
Similar result

m = 0.25
itg 57/80(0.71) 17/40(0.43) 40/40(1.00)
orig 64/80(0.80) 29/40(0.73) 35/40(0.88)
resplit 136/177(0.77) 23/36(0.64) 113/141(0.80)
similar result

m = 10 * double.Epsilon
itg 44/80(0.55) 40/40(1.00) 4/40(0.10)
orig 64/80(0.80) 29/40(0.73) 35/40(0.88)
resplit 137/177(0.77) 23/36(0.64) 114/141(0.81)
similar result

My conclusion is that the data is sampled random enough that more training data does little to improve overal score.