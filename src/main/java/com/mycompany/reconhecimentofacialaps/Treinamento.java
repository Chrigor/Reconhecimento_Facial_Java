/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.reconhecimentofacialaps;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_imgproc.putText;

/**
 *
 * @author chrig
 */
public class Treinamento {

    public static void main(String[] args) {
        File diretorio = new File("src\\main\\java\\fotos");
        FilenameFilter filtroImagem = new FilenameFilter() {

            @Override
            public boolean accept(File file, String string) {
                return string.endsWith(".jpg") || string.endsWith(".gif") || string.endsWith(".png");
            }
        };

        File[] arq = diretorio.listFiles(filtroImagem);

        MatVector fotos = new MatVector(arq.length); // vetor de imagem
        Mat rotulos = new Mat(arq.length, 1, CV_32SC1);
        IntBuffer rotulosBuffer = rotulos.createBuffer();
        int contador = 0;

        for (File file : arq) {
            Mat foto = imread(file.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE); // carrega img em escala cinza já
            int classe = Integer.parseInt(file.getName().split("\\.")[1]);
            //System.out.println(classe);

            resize(foto, foto, new Size(160, 160));
            fotos.put(contador, foto);
            rotulosBuffer.put(contador, classe);
            contador++;
        }

        FaceRecognizer reconhecedor = opencv_face.LBPHFaceRecognizer.create();
        reconhecedor.train(fotos, rotulos);
        reconhecedor.save("src\\main\\java\\recursos\\classificadorLBPH.yml");
        
    }

}
