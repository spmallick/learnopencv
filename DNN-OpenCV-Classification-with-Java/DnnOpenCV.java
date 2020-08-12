import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.opencv.core.CvType;


public class DnnOpenCV {
    private static final int TARGET_IMG_WIDTH = 224;
    private static final int TARGET_IMG_HEIGHT = 224;

    private static final double SCALE_FACTOR = 1 / 255.0;

    private static final String IMAGENET_CLASSES = "imagenet_classes.txt";
    private static final String MODEL_PATH = "models/pytorch_mobilenet.onnx";


    public static ArrayList<String> getImgLabels(String imgLabelsFilePath) throws IOException {
        ArrayList<String> imgLabels;
        try (Stream<String> lines = Files.lines(Paths.get(imgLabelsFilePath))) {
            imgLabels = lines.collect(Collectors.toCollection(ArrayList::new));
        }
        return imgLabels;
    }

    public static Mat centerCrop(Mat inputImage) {
        int y1 = Math.round((inputImage.rows() - TARGET_IMG_HEIGHT) / 2);
        int y2 = Math.round(y1 + TARGET_IMG_HEIGHT);
        int x1 = Math.round((inputImage.cols() - TARGET_IMG_WIDTH) / 2);
        int x2 = Math.round(x1 + TARGET_IMG_WIDTH);

        Rect centerRect = new Rect(x1, y1, (x2 - x1), (y2 - y1));
        Mat croppedImage = new Mat(inputImage, centerRect);

        return croppedImage;
    }

    public static Mat getPreprocessedImage(String imagePath) {
        // define mean and standard deviation
        Scalar mean = new Scalar(0.485, 0.456, 0.406);
        Scalar std = new Scalar(0.229, 0.224, 0.225);

        // get the image from the internal resource folder
        Mat image = Imgcodecs.imread(imagePath);

        // resize input image
        Imgproc.resize(image, image, new Size(256, 256));

        // create empty Mat images for float conversions
        Mat imgFloat = new Mat(image.rows(), image.cols(), CvType.CV_32FC3);

        // convert input image to float type
        image.convertTo(imgFloat, CvType.CV_32FC3, SCALE_FACTOR);

        // crop input image
        imgFloat = centerCrop(imgFloat);

        // prepare DNN input
        Mat blob = Dnn.blobFromImage(
                imgFloat,
                1.0, /* default scalefactor */
                new Size(TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT), /* target size */
                mean,  /* mean */
                true, /* swapRB */
                false /* crop */
        );

        // divide on std
        Core.divide(blob, std, blob);

        return blob;
    }

    public static void getPredictedClass(Mat classificationResult) {
        ArrayList<String> imgLabels = new ArrayList<String>();
        try {
            imgLabels = getImgLabels(IMAGENET_CLASSES);
        } catch (IOException ex) {

        }
        // obtain max prediction result
        Core.MinMaxLocResult mm = Core.minMaxLoc(classificationResult);
        double maxValIndex = mm.maxLoc.x;
        System.out.println("Predicted Class: " + imgLabels.get((int) maxValIndex));
    }

    public static void main(String[] args) {
        String imageLocation = "images/coffee.jpg";

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // read and process the input image
        Mat inputBlob = DnnOpenCV.getPreprocessedImage(imageLocation);

        // read generated ONNX model into org.opencv.dnn.Net object
        Net dnnNet = Dnn.readNetFromONNX(DnnOpenCV.MODEL_PATH);
        System.out.println("DNN from ONNX was successfully loaded!");

        // set OpenCV model input
        dnnNet.setInput(inputBlob);

        // provide inference
        Mat classification = dnnNet.forward();

        // decode classification results
        DnnOpenCV.getPredictedClass(classification);
    }
}