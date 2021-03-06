# AttentionClipper
## What Is This
![](images/attention_clipper.png)

It retrieves `sensor_msgs/CameraInfo` and publish `sensor_msgs/CameraInfo` with ROI filled and
retrieves `sensor_msgs/PointCloud2` and publish `pcl_msgs/PointIndices`.

You can specify the pose and size of the interest bounding box and jsk\_pcl/AttentionClipper returns ROI
to see the object.

## Subscribing Topic
* `~input` (`sensor_msgs/CameraInfo`)

  Original camera info.

* `~input/points` (`sensor_msgs/PointCloud2`)

  Original pointcloud.
* `~input/pose` (`geometry_msgs/PoseStamped`)
* `~input/box` (`jsk_recognition_msgs/BoundingBox`)
  Specify the pose of the bounding box. Timestamp will be ignored and camera info's timestamp will be used. If you use `~input/box`, you can change the size of attention region. There callbacks are only enabled if `~use_multiple_attention` is false.

* `~input/pose_array` (`geometry_msgs/PoseArray`)
* `~input/box_array` (`jsk_recognition_msgs/BoundingBoxArray`)
  It's an array version of `~input/pose` and `~input/box`. There callbacks are only enabled if `~use_multiple_attention` is true.

## Publishing Topic
* `~output` (`sensor_msgs/CameraInfo`)

  This camera info is same with `~input` except for roi field.
  (only when `~input` is published.)

* `~output/mask` (`sensor_msgs/Image`)

  Mask image to mask the regions of specified interest.
  (only when `~input` is published.)

* `~output/point_indices` (`pcl_msgs/PointIndices`)

  Indices of `~input/points` which are inside of interest regions.

## Parameter
* `~use_multiple_attention` (Boolean, default: `False`)

  If you want to enable multiple attentions, please set this variable True.

* `~dimension_x` (Double, default: `0.1`)
* `~dimension_y` (Double, default: `0.1`)
* `~dimension_z` (Double, default: `0.1`)

  Size of bounding box. Available only if `~use_multiple_attention` is false.

* `~frame_id` (String, default: `base_link`)

  Frame id of attention region. Available only if `~use_multiple_attention` is false.

* `~initial_pos` (Array of double, default: `None`):

  Initial pose of interesting region. Available only if `~use_multiple_attention` is false.

* `~initial_rot` (Array of double, default: `None`):

  Initial orientation of interesting region. The value should be represented in
  [roll, pitch, yaw]. Available only if `~use_multiple_attention` is false.

* `~initial_pos_list` (Array of array of double, default: `None`)
* `~initial_rot_list` (Array of array of double, default: `None`)
* `~frame_id_list` (Array of string, default: `None`)
* `~dimensions` (Array of array of double, default: `None`)
* `~prefixes` (Array of string, default: `None`)

  Position, Rotation, frame id ,prefix and Dimensions of multiple attention regions respectively.
  `~iniital_pos_list` should follow `[[x, y, z], ...]`,
  `~initial_rot_list` should follow `[[rx, ry, rz], ...]` and
  `~dimensions` should follow `[[x, y, z], ...]`.
  `~prefixes` `[prefix1, prefix2, ...]`. These prefixes will add to the /point_indices and advertise
  Available only if `~use_multiple_attention` is true.
* `~negative` (Boolean, default: `False`)

  Publish points which are not inside of attention regions if this parameter is true.
