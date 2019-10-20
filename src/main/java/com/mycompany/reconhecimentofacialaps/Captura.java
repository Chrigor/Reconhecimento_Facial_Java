package com.mycompany.reconhecimentofacialaps;

import java.awt.event.KeyEvent;
import java.util.Scanner;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_PLAIN;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
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

        String[] grupoList = {"", "pessoa"};

        OpenCVFrameConverter.ToMat conversorFrameToMat = new OpenCVFrameConverter.ToMat();// conversão para matriz
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0); // número do dispositivo (camera)

        Size tamMin = new Size(160, 160);
        Size tamMax = new Size(500, 500);

        camera.start(); // inicia captura das webCam

        CascadeClassifier detectorFaces = new CascadeClassifier("src\\main\\java\\recursos\\haarcascade.xml"); // arquivo treinado para deteccao de faces

        FaceRecognizer reconhecedor = opencv_face.LBPHFaceRecognizer.create();
        reconhecedor.read("src\\main\\java\\recursos\\classificadorLBPH.yml");

        //carregou o bang
        CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma()); // janela preview
        Frame frameCapturado = null;

        Mat imagemColorida = new Mat(); // atraves daq detecta face
        int quantidadeAmostras = 25; // media para um bom desempenho
        int amostra = 1;

        /*System.out.println("Digite seu ID: ");
        Scanner cad = new Scanner(System.in);
        int id = cad.nextInt();*/
        //int id = 0;
        int identificador = 0;

        while ((frameCapturado = camera.grab()) != null) { // enquanto estiver capturando coisa da webcam
            imagemColorida = conversorFrameToMat.convert(frameCapturado);

            Mat imagemCinza = new Mat();
            cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY); // conversor da imagem em cinza, pois trabalham melhor com cinza

            RectVector facesDetectadas = new RectVector();

            detectorFaces.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, tamMin, tamMax); //sizes = tamanhos de face min e max

            for (int i = 0; i < facesDetectadas.size(); i++) {
                Rect dadosFace = facesDetectadas.get(i); // retangulo
                rectangle(imagemColorida, dadosFace, new Scalar(0, 0, 255, 0)); // cor em rgb
                Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                resize(faceCapturada, faceCapturada, new Size(160, 160)); // imagens do mesmo tamanho

                IntPointer rotulo = new IntPointer(1);
                DoublePointer erro = new DoublePointer(1);

                reconhecedor.predict(faceCapturada, rotulo, erro);
                
                identificador = rotulo.get(0);

                String name;

                /*if (amostra <= quantidadeAmostras) {
                    imwrite("src\\main\\java\\fotos\\pessoa." + id + "." + amostra + ".jpg", faceCapturada);
                    System.out.println("Foto " + amostra + " capturada");
                    amostra++;
                } */
                //System.out.println("ERROR: " + erro.get(1));
                
                if (identificador == -1 || erro.get(0) > 390) {
                    name = "Desconhecido";
                    //erroAcesso = true;
                    //Caso seja identificado uma face conhecida pelo sistema
                } else {
                    //System.out.println(grupoList[identificador]);
                    name = "Teste " + grupoList[identificador] + ": identificando...";
                    //erroAcesso = false;
                }

                int x = Math.max(dadosFace.tl().x(), 0);
                int y = Math.max(dadosFace.tl().y(), 0);
                putText(imagemColorida, name, new opencv_core.Point(x, y), FONT_HERSHEY_PLAIN, 1.4, new Scalar(0, 255, 0, 0));
            }

            if (cFrame.isVisible()) { // se estiver visivel [Tela]
                cFrame.showImage(frameCapturado); // mostre a imagem na tela
            }

            /*if (amostra > quantidadeAmostras) {
                break;
            } */
        }

        // cFrame.dispose(); // liberar memoria
        //camera.stop(); // para captura
    }
}
