package Projects.TarangAI.backend;

import org.opencv.core.Core;

public class opencvTesting {
    public static void main(String[] args) {
        System.load("E:/lib/opencv/build/java/x64/opencv_java4110.dll"); // Ensure correct path
        System.out.println("OpenCV loaded successfully! Version: " + Core.VERSION);
    }
}
