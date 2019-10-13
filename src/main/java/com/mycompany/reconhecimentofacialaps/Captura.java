package com.mycompany.reconhecimentofacialaps;

import java.awt.event.KeyEvent;
import java.util.Scanner;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

public class Captura {

    public static void main(String[] args) throws FrameGrabber.Exception {
        KeyEvent tecla = null;

        OpenCVFrameConverter.ToMat conversorFrameToMat = new OpenCVFrameConverter.ToMat();// conversão para matriz
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0); // número do dispositivo (camera)

        Size tamMin = new Size(160, 160);
        Size tamMax = new Size(500, 500);

        camera.start(); // inicia captura das webCam

        CascadeClassifier detectorFaces = new CascadeClassifier("src\\main\\java\\recursos\\haarcascade.xml"); // arquivo treinado para deteccao de faces

        CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma()); // janela preview
        Frame frameCapturado = null;

        Mat imagemColorida = new Mat(); // atraves daq detecta face
        int quantidadeAmostras = 25; // media para um bom desempenho
        int amostra = 1;

        System.out.println("Digite seu ID: ");
        Scanner cad = new Scanner(System.in);
        int id = cad.nextInt();

        while ((frameCapturado = camera.grab()) != null) { // enquanto estiver capturando coisa da webcam
            imagemColorida = conversorFrameToMat.convert(frameCapturado);

            Mat imagemCinza = new Mat();
            cvtColor(imagemColorida, imagemCinza, 2); // conversor da imagem em cinza, pois trabalham melhor com cinza

            RectVector facesDetectadas = new RectVector();

            detectorFaces.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, tamMin, tamMax); //sizes = tamanhos de face min e max

            for (int i = 0; i < facesDetectadas.size(); i++) {
                Rect dadosFace = facesDetectadas.get(0); // retangulo
                rectangle(imagemColorida, dadosFace, new Scalar(0, 0, 255, 0)); // cor em rgb
                Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                resize(faceCapturada, faceCapturada, new Size(160, 160)); // imagens do mesmo tamanho

                if (amostra <= quantidadeAmostras) {
                    imwrite("src\\main\\java\\fotos\\pessoa." + id + "." + amostra + ".jpg", faceCapturada);
                    System.out.println("Foto " + amostra + " capturada");
                    amostra++;
                }

            }

            if (cFrame.isVisible()) { // se estiver visivel [Tela]
                cFrame.showImage(frameCapturado); // mostre a imagem na tela
            }

            if (amostra > quantidadeAmostras) {
                break;
            }
        }

        cFrame.dispose(); // liberar memoria

        camera.stop(); // para captura
    }
}
