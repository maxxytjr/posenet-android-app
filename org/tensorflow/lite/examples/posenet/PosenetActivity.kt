/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.posenet

import android.Manifest
import android.app.AlertDialog
import android.app.Dialog
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.Rect
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.support.v4.app.ActivityCompat
import android.support.v4.app.DialogFragment
import android.support.v4.app.Fragment
import android.support.v4.content.ContextCompat
import android.util.Log
import android.util.Size
import android.util.SparseIntArray
import android.view.LayoutInflater
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Toast
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.acos
import kotlin.math.atan
import kotlin.math.PI
import kotlin.comparisons.minOf
import kotlin.comparisons.maxOf
import org.tensorflow.lite.examples.posenet.lib.BodyPart
import org.tensorflow.lite.examples.posenet.lib.Person
import org.tensorflow.lite.examples.posenet.lib.Posenet
import java.lang.Exception

class PosenetActivity :
  Fragment(),
  ActivityCompat.OnRequestPermissionsResultCallback {

  /** List of body joints that should be connected.    */
  private val bodyJoints = listOf(
    Pair(BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW),
    Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER),
    Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
    Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
    Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
    Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
    Pair(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
    Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER),
    Pair(BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
    Pair(BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
    Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
    Pair(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)
  )

  /** Threshold for confidence score. */
  private val minConfidence = 0.65

  // time elapsed
  private var timeStart = System.currentTimeMillis()
  private var timeElapsed = System.currentTimeMillis()

  /** Radius of circle used to draw keypoints.  */
  private val circleRadius = 9.0f

  /** Paint class holds the style and color information to draw geometries,text and bitmaps. */
  private var kpPaint = Paint()
  // paint class instance for pose lines
  private var linePaint = Paint()
  // paint class instance for text
  private var textPaint = Paint()
  // for writing coordinates of each detected keypoint
  private var bodyPartCoordTextPaint = Paint()
  // for plank text
  private var plankTextPaint = Paint()
  // for squat text
  private var squatTextPaint = Paint()

  //accumulated plank time
  private var plank_begin = System.currentTimeMillis()
  // accumulated squat reps
  private var squat_reps = 0
  private var squat_down_flag = false
  private var knees_not_caving_flag = true
  private var knees_caving_count = 0
  private var no_forward_lean_flag = true
  private var knees_not_over_toes_flag = true
  private var forward_lean_count = 0
  private var knees_over_toes_count = 0

  /** A shape for extracting frame data.   */
  private val PREVIEW_WIDTH = 640
  private val PREVIEW_HEIGHT = 480

  /** An object for the Posenet library.    */
  private lateinit var posenet: Posenet

  /** ID of the current [CameraDevice].   */
  private var cameraId: String? = null

  /** A [SurfaceView] for camera preview.   */
  private var surfaceView: SurfaceView? = null

  /** A [CameraCaptureSession] for camera preview.   */
  private var captureSession: CameraCaptureSession? = null

  /** A reference to the opened [CameraDevice].    */
  private var cameraDevice: CameraDevice? = null

  /** The [android.util.Size] of camera preview.  */
  private var previewSize: Size? = null

  /** The [android.util.Size.getWidth] of camera preview. */
  private var previewWidth = 0

  /** The [android.util.Size.getHeight] of camera preview.  */
  private var previewHeight = 0

  /** A counter to keep count of total frames before performing inference.  */
  private var frameCounter = 0

  // frame counter to keep track of total number of frames, to track fps
  private var numFrames = 0

  // inference frame rate; how many frames before performing inference
  private var inferenceRate = 3

  // for fps
  private var fps = 0

  /** An IntArray to save image data in ARGB8888 format  */
  private lateinit var rgbBytes: IntArray

  /** A ByteArray to save image data in YUV format  */
  private var yuvBytes = arrayOfNulls<ByteArray>(3)

  /** An additional thread for running tasks that shouldn't block the UI.   */
  private var backgroundThread: HandlerThread? = null

  /** A [Handler] for running tasks in the background.    */
  private var backgroundHandler: Handler? = null

  /** An [ImageReader] that handles preview frame capture.   */
  private var imageReader: ImageReader? = null

  /** [CaptureRequest.Builder] for the camera preview   */
  private var previewRequestBuilder: CaptureRequest.Builder? = null

  /** [CaptureRequest] generated by [.previewRequestBuilder   */
  private var previewRequest: CaptureRequest? = null

  /** A [Semaphore] to prevent the app from exiting before closing the camera.    */
  private val cameraOpenCloseLock = Semaphore(1)

  /** Whether the current camera device supports Flash or not.    */
  private var flashSupported = false

  /** Orientation of the camera sensor.   */
  private var sensorOrientation: Int? = null

  /** Abstract interface to someone holding a display surface.    */
  private var surfaceHolder: SurfaceHolder? = null

  /** [CameraDevice.StateCallback] is called when [CameraDevice] changes its state.   */
  private val stateCallback = object : CameraDevice.StateCallback() {

    override fun onOpened(cameraDevice: CameraDevice) {
      cameraOpenCloseLock.release()
      this@PosenetActivity.cameraDevice = cameraDevice
      createCameraPreviewSession()
    }

    override fun onDisconnected(cameraDevice: CameraDevice) {
      cameraOpenCloseLock.release()
      cameraDevice.close()
      this@PosenetActivity.cameraDevice = null
    }

    override fun onError(cameraDevice: CameraDevice, error: Int) {
      onDisconnected(cameraDevice)
      this@PosenetActivity.activity?.finish()
    }
  }

  /**
   * A [CameraCaptureSession.CaptureCallback] that handles events related to JPEG capture.
   */
  private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
    override fun onCaptureProgressed(
      session: CameraCaptureSession,
      request: CaptureRequest,
      partialResult: CaptureResult
    ) {
    }

    override fun onCaptureCompleted(
      session: CameraCaptureSession,
      request: CaptureRequest,
      result: TotalCaptureResult
    ) {
    }
  }

  /**
   * Shows a [Toast] on the UI thread.
   *
   * @param text The message to show
   */
  private fun showToast(text: String) {
    val activity = activity
    activity?.runOnUiThread { Toast.makeText(activity, text, Toast.LENGTH_SHORT).show() }
  }

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? = inflater.inflate(R.layout.tfe_pn_activity_posenet, container, false)

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    surfaceView = view.findViewById(R.id.surfaceView)
    surfaceHolder = surfaceView!!.holder
  }

  override fun onResume() {
    super.onResume()
    startBackgroundThread()
  }

  override fun onStart() {
    super.onStart()
    openCamera()
    posenet = Posenet(this.context!!)
  }

  override fun onPause() {
    closeCamera()
    stopBackgroundThread()
    super.onPause()
  }

  override fun onDestroy() {
    super.onDestroy()
    posenet.close()
  }

  private fun requestCameraPermission() {
    if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
      ConfirmationDialog().show(childFragmentManager, FRAGMENT_DIALOG)
    } else {
      requestPermissions(arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA_PERMISSION)
    }
  }

  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<String>,
    grantResults: IntArray
  ) {
    if (requestCode == REQUEST_CAMERA_PERMISSION) {
      if (allPermissionsGranted(grantResults)) {
        ErrorDialog.newInstance(getString(R.string.tfe_pn_request_permission))
          .show(childFragmentManager, FRAGMENT_DIALOG)
      }
    } else {
      super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }
  }

  private fun allPermissionsGranted(grantResults: IntArray) = grantResults.all {
    it == PackageManager.PERMISSION_GRANTED
  }

  /**
   * Sets up member variables related to camera.
   */
  private fun setUpCameraOutputs() {

    val activity = activity
    val manager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    try {
      for (cameraId in manager.cameraIdList) {
        val characteristics = manager.getCameraCharacteristics(cameraId)

        // We don't use a front facing camera in this sample.
        val cameraDirection = characteristics.get(CameraCharacteristics.LENS_FACING)
        if (cameraDirection != null &&
          cameraDirection == CameraCharacteristics.LENS_FACING_FRONT
        ) {
          continue
        }

        previewSize = Size(PREVIEW_WIDTH, PREVIEW_HEIGHT)

        imageReader = ImageReader.newInstance(
          PREVIEW_WIDTH, PREVIEW_HEIGHT,
          ImageFormat.YUV_420_888, /*maxImages*/ 2
        )

        sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)!!

        previewHeight = previewSize!!.height
        previewWidth = previewSize!!.width

        // Initialize the storage bitmaps once when the resolution is known.
        rgbBytes = IntArray(previewWidth * previewHeight)

        // Check if the flash is supported.
        flashSupported =
          characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true

        this.cameraId = cameraId

        // We've found a viable camera and finished setting up member variables,
        // so we don't need to iterate through other available cameras.
        return
      }
    } catch (e: CameraAccessException) {
      Log.e(TAG, e.toString())
    } catch (e: NullPointerException) {
      // Currently an NPE is thrown when the Camera2API is used but not supported on the
      // device this code runs.
      ErrorDialog.newInstance(getString(R.string.tfe_pn_camera_error))
        .show(childFragmentManager, FRAGMENT_DIALOG)
    }
  }

  /**
   * Opens the camera specified by [PosenetActivity.cameraId].
   */
  private fun openCamera() {
    val permissionCamera = ContextCompat.checkSelfPermission(activity!!, Manifest.permission.CAMERA)
    if (permissionCamera != PackageManager.PERMISSION_GRANTED) {
      requestCameraPermission()
    }
    setUpCameraOutputs()
    val manager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    try {
      // Wait for camera to open - 2.5 seconds is sufficient
      if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
        throw RuntimeException("Time out waiting to lock camera opening.")
      }
      manager.openCamera(cameraId!!, stateCallback, backgroundHandler)
    } catch (e: CameraAccessException) {
      Log.e(TAG, e.toString())
    } catch (e: InterruptedException) {
      throw RuntimeException("Interrupted while trying to lock camera opening.", e)
    }
  }

  /**
   * Closes the current [CameraDevice].
   */
  private fun closeCamera() {
    if (captureSession == null) {
      return
    }

    try {
      cameraOpenCloseLock.acquire()
      captureSession!!.close()
      captureSession = null
      cameraDevice!!.close()
      cameraDevice = null
      imageReader!!.close()
      imageReader = null
    } catch (e: InterruptedException) {
      throw RuntimeException("Interrupted while trying to lock camera closing.", e)
    } finally {
      cameraOpenCloseLock.release()
    }
  }

  /**
   * Starts a background thread and its [Handler].
   */
  private fun startBackgroundThread() {
    backgroundThread = HandlerThread("imageAvailableListener").also { it.start() }
    backgroundHandler = Handler(backgroundThread!!.looper)
  }

  /**
   * Stops the background thread and its [Handler].
   */
  private fun stopBackgroundThread() {
    backgroundThread?.quitSafely()
    try {
      backgroundThread?.join()
      backgroundThread = null
      backgroundHandler = null
    } catch (e: InterruptedException) {
      Log.e(TAG, e.toString())
    }
  }

  /** Fill the yuvBytes with data from image planes.   */
  private fun fillBytes(planes: Array<Image.Plane>, yuvBytes: Array<ByteArray?>) {
    // Row stride is the total number of bytes occupied in memory by a row of an image.
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (i in planes.indices) {
      val buffer = planes[i].buffer
      if (yuvBytes[i] == null) {
        yuvBytes[i] = ByteArray(buffer.capacity())
      }
      buffer.get(yuvBytes[i]!!)
    }
  }

  /** A [OnImageAvailableListener] to receive frames as they are available.  */
  private var imageAvailableListener = object : OnImageAvailableListener {
    override fun onImageAvailable(imageReader: ImageReader) {
      // We need wait until we have some size from onPreviewSizeChosen
      if (previewWidth == 0 || previewHeight == 0) {
        return
      }

      val image = imageReader.acquireLatestImage() ?: return
      fillBytes(image.planes, yuvBytes)

      ImageUtils.convertYUV420ToARGB8888(
        yuvBytes[0]!!,
        yuvBytes[1]!!,
        yuvBytes[2]!!,
        previewWidth,
        previewHeight,
        /*yRowStride=*/ image.planes[0].rowStride,
        /*uvRowStride=*/ image.planes[1].rowStride,
        /*uvPixelStride=*/ image.planes[1].pixelStride,
        rgbBytes
      )

      // Create bitmap from int array
      val imageBitmap = Bitmap.createBitmap(
        rgbBytes, previewWidth, previewHeight,
        Bitmap.Config.ARGB_8888
      )

      // Create rotated version for portrait display
      val rotateMatrix = Matrix()
      rotateMatrix.postRotate(90.0f)

      val rotatedBitmap = Bitmap.createBitmap(
        imageBitmap, 0, 0, previewWidth, previewHeight,
        rotateMatrix, true
      )
      image.close()

      // Process an image for analysis at a specified frame rate
      frameCounter = (frameCounter + 1) % inferenceRate
      if (frameCounter == 0) {
        processImage(rotatedBitmap)
      }

      // update number of frames
      numFrames += 1

    }
  }

  /** Crop Bitmap to maintain aspect ratio of model input.   */
  private fun cropBitmap(bitmap: Bitmap): Bitmap {
    val bitmapRatio = bitmap.height.toFloat() / bitmap.width
    val modelInputRatio = MODEL_HEIGHT.toFloat() / MODEL_WIDTH
    var croppedBitmap = bitmap

    // Acceptable difference between the modelInputRatio and bitmapRatio to skip cropping.
    val maxDifference = 1e-5

    // Checks if the bitmap has similar aspect ratio as the required model input.
    when {
      abs(modelInputRatio - bitmapRatio) < maxDifference -> return croppedBitmap
      modelInputRatio < bitmapRatio -> {
        // New image is taller so we are height constrained.
        val cropHeight = bitmap.height - (bitmap.width.toFloat() / modelInputRatio)
        croppedBitmap = Bitmap.createBitmap(
          bitmap,
          0,
          (cropHeight / 2).toInt(),
          bitmap.width,
          (bitmap.height - cropHeight).toInt()
        )
      }
      else -> {
        val cropWidth = bitmap.width - (bitmap.height.toFloat() * modelInputRatio)
        croppedBitmap = Bitmap.createBitmap(
          bitmap,
          (cropWidth / 2).toInt(),
          0,
          (bitmap.width - cropWidth).toInt(),
          bitmap.height
        )
      }
    }
    return croppedBitmap
  }

  /** Set the paint color and size.    */
  private fun setTextPaint() {
    textPaint.color = Color.GREEN
    textPaint.textSize = 60.0f
    textPaint.strokeWidth = 8.0f
  }

  private fun setPlankTextPaint(){
    plankTextPaint.color = Color.RED
    plankTextPaint.textSize = 40.0f
  }

  private fun setSquatTextPaint(){
    squatTextPaint.color = Color.rgb(24, 172, 228)
    squatTextPaint.textSize = 40.0f
  }

  private fun setKpPaint(){
    kpPaint.color = Color.rgb(230, 15, 195)
    kpPaint.strokeWidth = 8.0f
  }

  private fun setPoseLinePaint(){
    linePaint.color = Color.rgb(20, 237, 220)
    linePaint.strokeWidth = 8.0f
  }

  private fun setBodyPartCoordPaint(){
    bodyPartCoordTextPaint.color = Color.rgb(252, 186, 3)
    bodyPartCoordTextPaint.textSize = 25.0f
    bodyPartCoordTextPaint.strokeWidth = 8.0f
  }

  private fun find_bodypart_xy(person: Person, part_idx: Int, widthRatio: Float, heightRatio: Float,
                               left: Int, top: Int): List<Float>{
    var x_y = listOf<Float>(0.0f, 0.0f)
    try {
      var bodyPart = person.keyPoints[part_idx]
      var position = bodyPart.position
      var bodyPartX: Float = position.x.toFloat() * widthRatio + left
      var bodyPartY: Float = position.y.toFloat() * heightRatio + top

      x_y = listOf(bodyPartX, bodyPartY)

    }catch (e: Exception){
      Log.i(
        "bodyPart",
        "Cannot find body part"
      )
    }
    return x_y
  }

  private fun get_euclidean_dist(pt1: List<Float>, pt2: List<Float>): Int{
    var euclidean_dist = sqrt((pt1[0] - pt2[0]).pow(2) + (pt1[1] - pt2[1]).pow(2)).toInt()
    return euclidean_dist
  }

  private fun get_angle(pt0: List<Float>, pt1: List<Float>, pt2: List<Float>): Int {
    /**
     * pt0: reference point from which we measure the angle between pt1 and pt2
     * pt1: point 1 of interest
     * pt2: point 2 of interest
     * return: angle between pt0-pt1, and pt0-pt2
     */
    try {
      // cosine rule
      var a2 = (pt2[0] - pt1[0]).pow(2) + (pt2[1] - pt1[1]).pow(2)
      var b2 = (pt2[0] - pt0[0]).pow(2) + (pt2[1] - pt0[1]).pow(2)
      var c2 = (pt1[0] - pt0[0]).pow(2) + (pt1[1] - pt0[1]).pow(2)

      var angle_deg = (acos((b2 + c2 - a2) / sqrt(4 * b2 * c2)) * 180.0/PI).toInt()

      return angle_deg

    } catch (e: Exception){
      return 0
    }
  }

  private fun get_angle_atan(pt1: List<Float>, pt2: List<Float>): Int {
    /**
     * pt1: point 1 of interest (top point)
     * pt2: point 2 of interest (bottom point)
     * return: angle between pt0-pt1, and pt0-pt2
     */
    try {

      var angle_deg = abs((atan((pt2[1] - pt1[1]) / (pt2[0] - pt1[0])) * 180.0/PI).toInt())

      return angle_deg

    } catch (e: Exception){
      return 90
    }
  }

    //--------------------------------Plank analyzer function----------------------------//
  private fun analyze_plank(canvas: Canvas, person: Person){
    /**
     * lh_le_ls_angle: angle between left hand, left elbow and left elbow/shoulder
     * rh_re_rs_angle: angle between right hand, right elbow and right elbow/shoulder
     * ll_lhip_la_angle: angle between left shoulder, left hip and left ankle
     * rl_rhip_ra: angle between right shoulder, right hip and right ankle
     * n_la_dist: distance from left ankle to nose
     * n_ra_dist: distance from right ankle to nose
     */

    val screenWidth: Int
    val screenHeight: Int
    val left: Int
    val top: Int

    if (canvas.height > canvas.width) {
      screenWidth = canvas.width
      screenHeight = canvas.width
      left = 0
      top = (canvas.height - canvas.width) / 2
    } else {
      screenWidth = canvas.height
      screenHeight = canvas.height
      left = (canvas.width - canvas.height) / 2
      top = 0
    }

    val widthRatio = screenWidth.toFloat() / MODEL_WIDTH
    val heightRatio = screenHeight.toFloat() / MODEL_HEIGHT

    var lh_le_ls_angle: Int
    var rh_re_rs_angle: Int
    var ls_lhip_la_angle: Int
    var rs_rhip_ra_angle: Int
//    var n_la_dist: Int
//    var n_ra_dist: Int

    var arm_angle_range = 60..110
    var hip_angle_range = 140..180
//    var la_ra_dist_range = 50..250

    lh_le_ls_angle = get_angle(find_bodypart_xy(person, 7, widthRatio, heightRatio, left, top),
      find_bodypart_xy(person, 5, widthRatio, heightRatio, left, top),
      find_bodypart_xy(person, 9, widthRatio, heightRatio, left, top))

    rh_re_rs_angle = get_angle(find_bodypart_xy(person, 8, widthRatio, heightRatio, left, top),
      find_bodypart_xy(person, 6, widthRatio, heightRatio, left, top),
      find_bodypart_xy(person, 10, widthRatio, heightRatio, left, top))

    ls_lhip_la_angle = get_angle(find_bodypart_xy(person, 11, widthRatio, heightRatio, left, top),
      find_bodypart_xy(person, 5, widthRatio, heightRatio, left, top),
      find_bodypart_xy(person, 15, widthRatio, heightRatio, left, top))

    rs_rhip_ra_angle = get_angle(find_bodypart_xy(person, 12, widthRatio, heightRatio, left, top),
      find_bodypart_xy(person, 6, widthRatio, heightRatio, left, top),
      find_bodypart_xy(person, 16, widthRatio, heightRatio, left, top))

//    n_la_dist = get_euclidean_dist(find_bodypart_xy(person, 0, widthRatio, heightRatio, left, top),
//    find_bodypart_xy(person, 15, widthRatio, heightRatio, left, top))
//
//    n_ra_dist = get_euclidean_dist(find_bodypart_xy(person, 0, widthRatio, heightRatio, left, top),
//    find_bodypart_xy(person, 16, widthRatio, heightRatio, left, top))

    canvas.drawText(
      "Left Arm angle: %d".format(lh_le_ls_angle),
      (15.0f * widthRatio),
      (20.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Right Arm angle: %d".format(rh_re_rs_angle),
      (15.0f * widthRatio),
      (30.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Left Hip Angle: %d".format(ls_lhip_la_angle),
      (15.0f * widthRatio),
      (40.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Right Hip Angle: %d".format(rs_rhip_ra_angle),
      (15.0f * widthRatio),
      (50.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    // if angle conditions are met, flash total duration under tension
    if((lh_le_ls_angle in arm_angle_range || rh_re_rs_angle in arm_angle_range)
      && (ls_lhip_la_angle in hip_angle_range || rs_rhip_ra_angle in hip_angle_range)
      //&& (n_la_dist in la_ra_dist_range || n_ra_dist in la_ra_dist_range)
    ){

      var plank_time_elapsed = System.currentTimeMillis() - plank_begin

      canvas.drawText(
        "Plank",
        (70.0f * widthRatio),
        (20.0f * heightRatio + top),
        plankTextPaint
      )

      canvas.drawText(
        "Time elapsed: %.2f s".format(plank_time_elapsed * 1.0f / 1000),
        (70.0f * widthRatio),
        (40.0f * heightRatio + top),
        plankTextPaint
      )

    }
  }


  //-----------------------------------Squat analyzer Front View-------------------------------//
  private fun analyze_squat_front(canvas: Canvas, person: Person){

    val screenWidth: Int
    val screenHeight: Int
    val left: Int
    val right: Int
    val top: Int
    val bottom: Int

    if (canvas.height > canvas.width) {
      screenWidth = canvas.width
      screenHeight = canvas.width
      left = 0
      top = (canvas.height - canvas.width) / 2
    } else {
      screenWidth = canvas.height
      screenHeight = canvas.height
      left = (canvas.width - canvas.height) / 2
      top = 0
    }
    right = left + screenWidth
    bottom = top + screenHeight

    val widthRatio = screenWidth.toFloat() / MODEL_WIDTH
    val heightRatio = screenHeight.toFloat() / MODEL_HEIGHT

    // Get xy coordinates of keypoints of interest (ankle, knee hip)
    var lhip_xy: List<Float> = find_bodypart_xy(person, 11, widthRatio, heightRatio, left, top)
    var rhip_xy: List<Float> = find_bodypart_xy(person, 12, widthRatio, heightRatio, left, top)

    var lknee_xy: List<Float> = find_bodypart_xy(person, 13, widthRatio, heightRatio, left, top)
    var rknee_xy: List<Float> = find_bodypart_xy(person, 14, widthRatio, heightRatio, left, top)

    var lankle_xy: List<Float> = find_bodypart_xy(person, 15, widthRatio, heightRatio, left, top)
    var rankle_xy: List<Float> = find_bodypart_xy(person, 16, widthRatio, heightRatio, left, top)

    // range of knee angles that satisfy good squat depth
    var squat_knee_angle_range = 20..100
    // range of knee angles that imply standing position
    var stand_knee_angle_range = 160..190

    // angle that left thigh makes with calf. To check for depth
    var lknee_angle: Int = get_angle(lknee_xy, lhip_xy, lankle_xy)
    // angle that right thigh makes with calf
    var rknee_angle: Int = get_angle(rknee_xy, rhip_xy, rankle_xy)

    // offsets of knee from hip (to check for caving of knees)
    var lknee_hip_offset: Int = (lknee_xy[0] - lhip_xy[0]).toInt()
    var rknee_hip_offset: Int = (rhip_xy[0] - rknee_xy[0]).toInt()

    canvas.drawText(
      "Left knee angle: %d".format(lknee_angle),
      (10.0f * widthRatio),
      (20.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Right knee angle: %d".format(rknee_angle),
      (10.0f * widthRatio),
      (30.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Left knee offset: %d".format(lknee_hip_offset),
      (10.0f * widthRatio),
      (40.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Right knee offset: %d".format(rknee_hip_offset),
      (10.0f * widthRatio),
      (50.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Knees caved-in count: %d".format(knees_caving_count),
      (10.0f * widthRatio),
      (60.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Squat reps: %d".format(squat_reps),
      (90.0f * widthRatio),
      (20.0f * heightRatio + top),
      squatTextPaint
    )

    // check if there are no instances of caving knees throughout motion
    if(lknee_hip_offset >= -5 && rknee_hip_offset >= -5){
      // if athlete achieves desired depth
      if (lknee_angle in squat_knee_angle_range && rknee_angle in squat_knee_angle_range){
        squatTextPaint.color = Color.GREEN
        canvas.drawText(
          "Good!",
          (70.0f * widthRatio),
          (40.0f * heightRatio + top),
          squatTextPaint
        )
        squat_down_flag = true
        knees_not_caving_flag = true
      }
    } else{
      // the moment caving knees are detected, flash on screen
      squatTextPaint.color = Color.RED
      canvas.drawText(
        "KNEES CAVING IN!",
        (70.0f * widthRatio),
        (40.0f * heightRatio + top),
        squatTextPaint
      )
      knees_not_caving_flag = false
      squat_down_flag = false
    }

    // if athlete is back to standing position
    if (lknee_angle in stand_knee_angle_range && rknee_angle in stand_knee_angle_range){
      // if athlete has achieved desired squat depth
      if(squat_down_flag == true){
        squat_down_flag = false
        squat_reps += 1
      }

      // if athlete was found to have knees caved in throughout motion
      if(knees_not_caving_flag == false){
        knees_caving_count += 1
        knees_not_caving_flag = true
      }

    }


  }

  //-----------------------------------Squat analyzer Side View-------------------------------//
  private fun analyze_squat_side(canvas: Canvas, person: Person){

    val screenWidth: Int
    val screenHeight: Int
    val left: Int
    val top: Int

    if (canvas.height > canvas.width) {
      screenWidth = canvas.width
      screenHeight = canvas.width
      left = 0
      top = (canvas.height - canvas.width) / 2
    } else {
      screenWidth = canvas.height
      screenHeight = canvas.height
      left = (canvas.width - canvas.height) / 2
      top = 0
    }

    val widthRatio = screenWidth.toFloat() / MODEL_WIDTH
    val heightRatio = screenHeight.toFloat() / MODEL_HEIGHT

    // Get xy coordinates of keypoints of interest (shoulder, ankle, knee, hip)
    var lshoulder_xy: List<Float> = find_bodypart_xy(person, 5, widthRatio, heightRatio, left, top)
    var rshoulder_xy: List<Float> = find_bodypart_xy(person, 6, widthRatio, heightRatio, left, top)

    var lhip_xy: List<Float> = find_bodypart_xy(person, 11, widthRatio, heightRatio, left, top)
    var rhip_xy: List<Float> = find_bodypart_xy(person, 12, widthRatio, heightRatio, left, top)

    var lknee_xy: List<Float> = find_bodypart_xy(person, 13, widthRatio, heightRatio, left, top)
    var rknee_xy: List<Float> = find_bodypart_xy(person, 14, widthRatio, heightRatio, left, top)

    var lankle_xy: List<Float> = find_bodypart_xy(person, 15, widthRatio, heightRatio, left, top)
    var rankle_xy: List<Float> = find_bodypart_xy(person, 16, widthRatio, heightRatio, left, top)

    // range of knee angles that satisfy good squat depth
    var squat_knee_angle_range = 20..100
    // range of knee angles that imply standing position
    var stand_knee_angle_range = 160..190
    // range of angles that imply knees are not over toes by too much
    var knee_ankle_angle_range = 40..105
    // range of angles implying suitable amount of forward lean
    var torso_angle_range = 35..200

    // angle that left thigh makes with calf. To check for depth
    var lknee_angle: Int = get_angle(lknee_xy, lhip_xy, lankle_xy)
    // angle that right thigh makes with calf
    var rknee_angle: Int = get_angle(rknee_xy, rhip_xy, rankle_xy)

    // angle that torso makes with thigh. Check for excessive forward lean
    var ltorso_angle: Int = get_angle(lhip_xy, lshoulder_xy, lknee_xy)
    var rtorso_angle: Int = get_angle(rhip_xy, rshoulder_xy, rknee_xy)

    // angle between left knee-ankle line and horizontal
    var lankle_angle: Int = get_angle_atan(lknee_xy, lankle_xy)
    // angle between right knee-ankle line and horizontal
    var rankle_angle: Int = get_angle_atan(rknee_xy, rankle_xy)

    // take the minimum angle amongst the 2; one side might only get detected from a side view
    var ankle_angle: Int = minOf(lankle_angle, rankle_angle)
    var torso_angle: Int = minOf(ltorso_angle, rtorso_angle)
    var knee_angle: Int = minOf(lknee_angle, rknee_angle)

    bodyPartCoordTextPaint.color = Color.RED
    canvas.drawText(
      "Ankle angle (left): %d".format(lankle_angle),
      (10.0f * widthRatio),
      (15.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Ankle angle (right): %d".format(rankle_angle),
      (10.0f * widthRatio),
      (30.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Torso angle (left): %d".format(ltorso_angle),
      (10.0f * widthRatio),
      (45.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Torso angle (right): %d".format(rtorso_angle),
      (10.0f * widthRatio),
      (60.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Knee angle (left): %d".format(lknee_angle),
      (10.0f * widthRatio),
      (75.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Knee angle (right): %d".format(rknee_angle),
      (10.0f * widthRatio),
      (90.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Excessive forward lean: %d".format(forward_lean_count),
      (10.0f * widthRatio),
      (105.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Knees over toes: %d".format(knees_over_toes_count),
      (10.0f * widthRatio),
      (120.0f * heightRatio + top),
      bodyPartCoordTextPaint
    )

    canvas.drawText(
      "Squat reps: %d".format(squat_reps),
      (80.0f * widthRatio),
      (20.0f * heightRatio + top),
      squatTextPaint
    )

    // if athlete achieves desirable squat form while there are no instances of knees over toes (70..105) and excessive forward lean
    if(ankle_angle in knee_ankle_angle_range && torso_angle in torso_angle_range){
      // check if good squat depth is achieved (20..100)
      if (knee_angle in squat_knee_angle_range){
        squatTextPaint.color = Color.GREEN
        canvas.drawText(
          "Good!",
          (80.0f * widthRatio),
          (40.0f * heightRatio + top),
          squatTextPaint
        )
        squat_down_flag = true
        knees_not_over_toes_flag = true
        no_forward_lean_flag = true
      }

    }

    if (ankle_angle !in knee_ankle_angle_range){
      // the moment knees are way over toes, flash on screen and set squat down flag to false (no count)
      squatTextPaint.color = Color.RED
      canvas.drawText(
        "KNEES OVER TOES!",
        (80.0f * widthRatio),
        (40.0f * heightRatio + top),
        squatTextPaint
      )
      knees_over_toes_count += 1
      knees_not_over_toes_flag = false
      squat_down_flag = false
    }

    if(torso_angle !in torso_angle_range){     // if excessive forward lean occurs while squatting down or up
      squatTextPaint.color = Color.RED
      canvas.drawText(
        "EXCESSIVE FORWARD LEAN!",
        (80.0f * widthRatio),
        (40.0f * heightRatio + top),
        squatTextPaint
      )
      forward_lean_count += 1
      no_forward_lean_flag = false
      squat_down_flag = false
    }

    // if athlete goes from valid squat position to standing position, increase rep count, reset flags
    if (squat_down_flag == true &&
      no_forward_lean_flag == true &&
      knees_not_over_toes_flag == true &&
      lknee_angle in stand_knee_angle_range
      && rknee_angle in stand_knee_angle_range){

      squat_down_flag = false
      no_forward_lean_flag = false
      knees_not_over_toes_flag = false

      squat_reps += 1
    }


  }

  /** Draw bitmap on Canvas.   */
  private fun draw(canvas: Canvas, person: Person, bitmap: Bitmap) {
    canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
    // Draw `bitmap` and `person` in square canvas.
    val screenWidth: Int
    val screenHeight: Int
    val left: Int
    val right: Int
    val top: Int
    val bottom: Int
    if (canvas.height > canvas.width) {
      screenWidth = canvas.width
      screenHeight = canvas.width
      left = 0
      top = (canvas.height - canvas.width) / 2
    } else {
      screenWidth = canvas.height
      screenHeight = canvas.height
      left = (canvas.width - canvas.height) / 2
      top = 0
    }
    right = left + screenWidth
    bottom = top + screenHeight

    // initialize the paint settings
    setTextPaint()
    setKpPaint()
    setPoseLinePaint()
    setBodyPartCoordPaint()
    setPlankTextPaint()
    setSquatTextPaint()

    canvas.drawBitmap(
      bitmap,
      Rect(0, 0, bitmap.width, bitmap.height),
      Rect(left, top, right, bottom),
      textPaint
    )

    val widthRatio = screenWidth.toFloat() / MODEL_WIDTH
    val heightRatio = screenHeight.toFloat() / MODEL_HEIGHT

    // Draw key points over the image.
    for (keyPoint in person.keyPoints) {
      if (keyPoint.score > minConfidence) {
        val position = keyPoint.position
        val adjustedX: Float = position.x.toFloat() * widthRatio + left
        val adjustedY: Float = position.y.toFloat() * heightRatio + top
        canvas.drawCircle(adjustedX, adjustedY, circleRadius, kpPaint)
      }
    }

    for (line in bodyJoints) {
      if (
        (person.keyPoints[line.first.ordinal].score > minConfidence) and
        (person.keyPoints[line.second.ordinal].score > minConfidence)
      ) {
        canvas.drawLine(
          person.keyPoints[line.first.ordinal].position.x.toFloat() * widthRatio + left,
          person.keyPoints[line.first.ordinal].position.y.toFloat() * heightRatio + top,
          person.keyPoints[line.second.ordinal].position.x.toFloat() * widthRatio + left,
          person.keyPoints[line.second.ordinal].position.y.toFloat() * heightRatio + top,
          linePaint
        )
      }
    }

    if(person.score > minConfidence){
      //TODO Associate this portion with a dropdown for user selection

//      // PLANK ANALYZER
//      analyze_plank(canvas, person)
        // SQUAT ANALYZER
        analyze_squat_front(canvas, person)
//      // SQUAT ANALYZER (SIDE VIEW)
//      analyze_squat_side(canvas, person)
    }

    //** Stuff below the camera's image **//
    canvas.drawText(
      "Score: %.2f".format(person.score),
      (15.0f * widthRatio),
      (20.0f * heightRatio + bottom),
      textPaint
    )
    canvas.drawText(
      "Device: %s".format(posenet.device),
      (15.0f * widthRatio),
      (40.0f * heightRatio + bottom),
      textPaint
    )
    canvas.drawText(
      "Time Elapsed: %.2f s".format(timeElapsed * 1.0f / 1_000),
      (15.0f * widthRatio),
      (60.0f * heightRatio + bottom),
      textPaint
    )

    canvas.drawText(
      "FPS: %.2f".format(numFrames / (timeElapsed * 1.0f / 1_000)),
      (15.0f * widthRatio),
      (80.0f * heightRatio + bottom),
      textPaint
    )

    // Draw!
    surfaceHolder!!.unlockCanvasAndPost(canvas)
  }

  /** Process image using Posenet library.   */
  private fun processImage(bitmap: Bitmap) {
    // Crop bitmap.
    val croppedBitmap = cropBitmap(bitmap)

    // Created scaled version of bitmap for model input.
    val scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, MODEL_WIDTH, MODEL_HEIGHT, true)

    // Perform inference.
    val person = posenet.estimateSinglePose(scaledBitmap)
    val canvas: Canvas = surfaceHolder!!.lockCanvas()
    draw(canvas, person, scaledBitmap)

    timeElapsed = System.currentTimeMillis() - timeStart
  }

  /**
   * Creates a new [CameraCaptureSession] for camera preview.
   */
  private fun createCameraPreviewSession() {
    try {

      // We capture images from preview in YUV format.
      imageReader = ImageReader.newInstance(
        previewSize!!.width, previewSize!!.height, ImageFormat.YUV_420_888, 2
      )
      imageReader!!.setOnImageAvailableListener(imageAvailableListener, backgroundHandler)

      // This is the surface we need to record images for processing.
      val recordingSurface = imageReader!!.surface

      // We set up a CaptureRequest.Builder with the output Surface.
      previewRequestBuilder = cameraDevice!!.createCaptureRequest(
        CameraDevice.TEMPLATE_PREVIEW
      )
      previewRequestBuilder!!.addTarget(recordingSurface)

      // Here, we create a CameraCaptureSession for camera preview.
      cameraDevice!!.createCaptureSession(
        listOf(recordingSurface),
        object : CameraCaptureSession.StateCallback() {
          override fun onConfigured(cameraCaptureSession: CameraCaptureSession) {
            // The camera is already closed
            if (cameraDevice == null) return

            // When the session is ready, we start displaying the preview.
            captureSession = cameraCaptureSession
            try {
              // Auto focus should be continuous for camera preview.
              previewRequestBuilder!!.set(
                CaptureRequest.CONTROL_AF_MODE,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
              )
              // Flash is automatically enabled when necessary.
              setAutoFlash(previewRequestBuilder!!)

              // Finally, we start displaying the camera preview.
              previewRequest = previewRequestBuilder!!.build()
              captureSession!!.setRepeatingRequest(
                previewRequest!!,
                captureCallback, backgroundHandler
              )
            } catch (e: CameraAccessException) {
              Log.e(TAG, e.toString())
            }
          }

          override fun onConfigureFailed(cameraCaptureSession: CameraCaptureSession) {
            showToast("Failed")
          }
        },
        null
      )
    } catch (e: CameraAccessException) {
      Log.e(TAG, e.toString())
    }
  }

  private fun setAutoFlash(requestBuilder: CaptureRequest.Builder) {
    if (flashSupported) {
      requestBuilder.set(
        CaptureRequest.CONTROL_AE_MODE,
        CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH
      )
    }
  }

  /**
   * Shows an error message dialog.
   */
  class ErrorDialog : DialogFragment() {

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
      AlertDialog.Builder(activity)
        .setMessage(arguments!!.getString(ARG_MESSAGE))
        .setPositiveButton(android.R.string.ok) { _, _ -> activity!!.finish() }
        .create()

    companion object {

      @JvmStatic
      private val ARG_MESSAGE = "message"

      @JvmStatic
      fun newInstance(message: String): ErrorDialog = ErrorDialog().apply {
        arguments = Bundle().apply { putString(ARG_MESSAGE, message) }
      }
    }
  }

  companion object {
    /**
     * Conversion from screen rotation to JPEG orientation.
     */
    private val ORIENTATIONS = SparseIntArray()
    private val FRAGMENT_DIALOG = "dialog"

    init {
      ORIENTATIONS.append(Surface.ROTATION_0, 90)
      ORIENTATIONS.append(Surface.ROTATION_90, 0)
      ORIENTATIONS.append(Surface.ROTATION_180, 270)
      ORIENTATIONS.append(Surface.ROTATION_270, 180)
    }

    /**
     * Tag for the [Log].
     */
    private const val TAG = "PosenetActivity"
  }
}
