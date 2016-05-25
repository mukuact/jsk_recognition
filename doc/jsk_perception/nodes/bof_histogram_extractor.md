# BofHistgramExtractor
![](images/bof_object_recognition.png)

make Bag of feature histgram from interest object's SIFT descripters

## Subscribing Topics

* `~bof_data`

  Filename of bag of features dictionary extracted by create\_bof\_dataset.py

* `~input` ( posedetectioni\_msgs.msg/Feature0D)

  SIFT descripters extracted from input image

* `~input/label` (sensor_msgs/Image)

  a labeled mask image of interest objects

## Publishing Topics

* `~output` (`jsk\_recognition\_msgs/VectorArray`)

  Bag of features histgram

## Parameters

* `~queue_size` (Int, default: `10`)

* `~approximate_sync` (Bool, default: `False`)

  Approximately synchronize `~input` and `~input/label` if it's true.
