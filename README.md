## Interpretable / Explainable RandomForest

This is an implementation of [Ando Saabas' treeinterpreter](http://blog.datadive.net/interpreting-random-forests/).

----

Usage:

```
val contributions = InterpretableRF.interpret(rfModel, dataset, sc, featureNamesArray)
    
contributions
  .map({ case (id, arr) => (id, "{" + arr.mkString(",") + "}")})
  .saveAsTextFile("out/contributions")
```
Gives the contributions:


```
(451,{0.11874332829942136,-0.0215166134401914,-4.149519988899469,-0.16957979571375603,0.22385938280675133,0.027965445013499957,0.8659203399385582,-0.12557875649961436})
(19,{0.0092574246383344,-2.4408338196274997,5.456102476426396,0.12388665932259912,0.1312976071353812,0.42083611277655975,0.28850726156209755,-0.23539879033861094})
...
```

----

This is what it can look like inside a UI (red means probability-decreasing for that value for this instance, green means probability-increasing):

![UI example](https://github.com/benoitparis/InterpretableRandomForest/raw/master/interface_example.png "UI example")


