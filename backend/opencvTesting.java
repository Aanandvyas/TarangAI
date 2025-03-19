package Projects.TarangAI.backend;

import org.opencv.core.Core;

public class opencvTesting {
    public static void main(String[] args) {
        // Ensure correct path
        System.load("E:/lib/opencv/build/java/x64/opencv_java4110.dll"); 
        System.out.println("OpenCV loaded successfully! Version: " + Core.VERSION);
    }
}
