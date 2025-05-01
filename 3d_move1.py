import depthai as dai
import cv2
import numpy as np

pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
depth = pipeline.create(dai.node.StereoDepth)
xout_depth = pipeline.create(dai.node.XLinkOut)

xout_depth.setStreamName("depth")
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
left.out.link(depth.left)
right.out.link(depth.right)
depth.depth.link(xout_depth.input)

with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    print("Presiona 'q' para salir")
    while True:
        depth_frame = depthQueue.get().getFrame()
        depth_frame_color = cv2.normalize(depth_frame, None_
