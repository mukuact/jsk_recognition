# MatchTemplate
![]()

## What's this?

 return (1st-val,1st-loc,2nd-val,2nd-loc)

## subscribe topic:

*  `~/reference`           ( `sensor_msgs/Image` )

  reference_image

*  `~/search`              ( `sensor_msgs/Image` )

  source_image

*  `~/set_reference_point` ( `geometry_msgs/PointStamped` )

  setting search area center to given point for simple point tracking

*  `~/set_search_rect`     ( `jsk_perception/Rect` )

  set ROI(search area) of source_image

## publish topic:
*  `~/current_template`    ( `sensor_msgs/Image` )
*  `~/esult`               ( `geometry_msgs/TransformStamped` )
*  `~/debug_image`         ( `sensor_msgs/Image` )

## Parameters

# you can set below options for each templates
   color space: mono8,bgr8,bgra8,hsv,hsva (OpenCV expects bgr)
   template size: tuple (x,y,width,height)
   search area:   tuple (x,y,width,height)
   search topic: (TODO)
   match method: 6type
