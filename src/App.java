import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import org.opencv.imgproc.Imgproc;

public class App {
    public static List<Mat> build_filters(){
        List<Mat> filters = new ArrayList<>();
        var step = Math.PI/15;
        Size ksize = new Size(35,35);
        
        double[] range = IntStream.rangeClosed(0, (int)((Math.PI-0)/step)).mapToDouble(x -> x*(Math.PI/16)).toArray();
        for (var theta : range) {
            var kern = Imgproc.getGaborKernel(ksize, 4.3, theta, 10.0, 0.5,0,CvType.CV_32F);
            //Imgproc.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype)
            //ksize Size of the filter returned.
            //sigma Standard deviation of the gaussian envelope.
            //theta Orientation of the normal to the parallel stripes of a Gabor function.
            //lambda Wavelength of the sinusoidal factor.
            //gamma Spatial aspect ratio.
            //psi Phase offset.
            //ktype Type of filter coefficients. It can be CV_32F or CV_64F.
            double kf = 1.5*Core.sumElems(kern).val[0];
            Scalar sl = new Scalar(kf);
            Core.divide(kern, sl, kern);
            filters.add(kern);
        }
        

        return filters;
    }
    //thread this if possible 
    public static Mat process(Mat img, List<Mat> filters){
        Mat accum = Mat.zeros(img.size(),img.type());
        for (Mat kern : filters) {
            Mat fimg = new Mat();
            Imgproc.filter2D(img, fimg, CvType.CV_8UC3, kern);
            Core.max(fimg, accum, accum);
        }
        return accum;
    }
    public static void main(String[] args) throws Exception {
        String rootPath = "/images/";
        System.loadLibrary("lib/x64/opencv_java451");
        Imgcodecs imagImgcodecs = new Imgcodecs();
        Mat image = imagImgcodecs.imread(rootPath+"example.jpg");
        Size size = new Size(500,600);
        Imgproc.resize(image, image, size);
        Mat image_Gray = new Mat();
        // gray Scale
        Imgproc.cvtColor(image, image_Gray, Imgproc.COLOR_RGB2GRAY);
        imagImgcodecs.imwrite(rootPath+"exampleGrayScaleJava.png", image_Gray);
        // Equalize
        Mat image_CLAHE = new Mat();
        var clahe = Imgproc.createCLAHE(4);
        clahe.apply(image_Gray, image_CLAHE);
        imagImgcodecs.imwrite(rootPath+"exampleEqualizedJava.png", image_CLAHE);
        // Gabor filtering
        List<Mat> filters = build_filters();
        Mat image_gabor = new Mat();
        image_gabor = process(image_CLAHE,filters);
        imagImgcodecs.imwrite(rootPath+"exampleGaborJava.png", image_gabor);
        // Normalize image
        Mat image_Norm = Mat.zeros(image_gabor.size(),image_gabor.type());
        Core.normalize(image_gabor, image_Norm, 50, 155, 32);//32 = NORM_MINMAX
        imagImgcodecs.imwrite(rootPath+"exampleNormalizedJava.png", image_Norm);
        // Gausian C
        Mat Gausian_C_img = new Mat();
        Imgproc.adaptiveThreshold(image_Norm, Gausian_C_img, 255,  Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 25, 4);
        imagImgcodecs.imwrite(rootPath+"exampleGausianCJava.png", Gausian_C_img);
    }
}