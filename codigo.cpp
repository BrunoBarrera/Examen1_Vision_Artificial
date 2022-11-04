#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <cmath>

using namespace cv;
using namespace std;

int Convolution(int i, int j,double** kernel,Mat imagen) {
	int m = 0;
	int n = 0;
	float pixel = 0;
	float suma = 0;
	for (m = -1; m <= 1;m++) {
		for (n = -1; n <= 1; n++) {
			pixel = imagen.at<uchar>(Point(i+m, j+n));
			suma = suma + pixel * kernel[m + 1][n + 1];
		}
	}
	return abs(suma);
}

Mat RGB2Gray(Mat imagen, Mat Gray, int fila_original, int columna_original) {
	int i = 0;
	int j = 0;
	double azul = 0;
	double verde = 0;
	double rojo = 0;
	double pixel = 0;

	for (i = 0; i < fila_original; i++) {
		for (j = 0; j < columna_original; j++) {
			azul = imagen.at<Vec3b>(Point(i, j)).val[0];
			verde = imagen.at<Vec3b>(Point(i, j)).val[1];
			rojo = imagen.at<Vec3b>(Point(i, j)).val[2];
			pixel = (azul * 0.114 + verde * 0.587 + rojo * 0.299);
			Gray.at<uchar>(Point(i, j)) = uchar(pixel);
		}
	}
	return Gray;
}

Mat LlenadoCeros(Mat copia, int fila_original, int columna_original, int filas_copia, int columnas_copia, int filas_limite, int columnas_limite) {
	int i = 0;
	int j = 0;

	for (i = 0; i < filas_copia; i++) {
		for (j = 0; j < columnas_copia; j++) {
			if (i < filas_limite || i>fila_original + filas_limite) {
				copia.at<uchar>(Point(j, i)) = 0;
			}
			if (j < columnas_limite || j > columna_original + columnas_limite) {
				copia.at<uchar>(Point(j, i)) = 0;
			}
		}
	}
	return copia;
}

float Convolusion(Mat Gray, int extremos, double** kernel, int x, int y) {
	int i = 0;
	int j = 0;
	float valorPix = 0;

	for (i = -extremos; i <= extremos; i++) {
		for (j = -extremos; j <= extremos; j++) {
			float valor_kernel = kernel[i + extremos][j + extremos];
			int vecino_x = x + i;
			int vecino_y = y + j;
			float valor_img_original = 0;
			valor_img_original = Gray.at<uchar>(Point(vecino_x, vecino_y));
			valorPix = valorPix + (valor_kernel * valor_img_original);
		}
	}
	return valorPix;
}
double** InicializaciónKernel(double** kernel, int size_kernel, int sigma, int filas_limite, int columnas_limite) {
	int i = 0;
	int j = 0;
	int x = 0;
	int y = 0;
	double pixel_kernel = 0;
	x = -(filas_limite);
	y = columnas_limite;

	for (i = 0; i < size_kernel; i++) {
		kernel[i] = new double[size_kernel];
		for (j = 0; j < size_kernel; j++) {
			pixel_kernel = (1 / (2 * (3.1416) * (pow(sigma, 2))) * (exp(-(pow(x, 2) + (pow(y, 2))) / (2 * (pow(sigma, 2))))));
			kernel[i][j] = pixel_kernel;
			x = x + 1;
		}
		x = -(filas_limite);
		y = y - 1;
	}
	return kernel;
}

Mat AjusteBordes(Mat imagen, Mat copia, int fila_original, int columna_original, int filas_limite, int columnas_limite) {
	int i = 0;
	int j = 0;
	double azul = 0;
	double verde = 0;
	double rojo = 0;
	double pixel = 0;

	for (i = 0; i < fila_original; i++) {
		for (j = 0; j < columna_original; j++) {
			azul = imagen.at<Vec3b>(Point(i, j)).val[0];
			verde = imagen.at<Vec3b>(Point(i, j)).val[1];
			rojo = imagen.at<Vec3b>(Point(i, j)).val[2];
			pixel = (azul * 0.114 + verde * 0.587 + rojo * 0.299);
			copia.at<uchar>(Point(i + columnas_limite, j + filas_limite)) = uchar(pixel);
		}
	}
	return copia;
}



Mat AplicaFiltroGauss(Mat Gray, Mat Gauss, double** kernel, int size_kernel, int fila_original, int columna_original) {
	int i = 0;
	int j = 0;
	int extremos = floor(size_kernel / 2);
	int l = 0;
	for (i = 0; i < fila_original; i++) {
		int k = 0;
		for (j = 0; j < columna_original; j++) {
			Gauss.at<uchar>(Point(l, k)) = uchar(Convolusion(Gray, extremos, kernel, i, j));
			k = k + 1;
		}
		l = l + 1;
	}
	return Gauss;
}

int main() {
	/**DECLARACION DE LAS VARIABLES GENERALES**/
	char NombreImagen[] = "C:/Users/Bruno/Downloads/lena.jpg";
	Mat imagen; //Matriz que contiene nuestra imagen sin importar el formato
	/****/

	/*LECTURA DE LA IMAGEN**/
	imagen = imread(NombreImagen);

	if (!imagen.data) {
		cout << "ERROR AL CARGAR LA IMAGEN: " << NombreImagen << endl;
		exit(1);
	}

	/**PROCESOS*/
	int fila_original = imagen.rows;
	int columna_original = imagen.cols;
	int size_kernel = 0;
	float sigma = 0;

	cout << "Ingresa kernel:\n";
	cin >> size_kernel;
	cout << "Ingresa sigma:\n";
	cin >> sigma;

	int filas_add = floor(size_kernel / 2);
	int filas_add_totales = filas_add * 2;
	int columnas_add_totales = filas_add_totales;
	int filas_copia = filas_add_totales + fila_original;
	int columnas_copia = columnas_add_totales + columna_original;
	Mat Gray(fila_original, columna_original, CV_8UC1);
	Mat copia(filas_copia, columnas_copia, CV_8UC1);

	Gray = RGB2Gray(imagen, Gray, fila_original, columna_original);

	int filas_limite = floor(size_kernel / 2);
	int columnas_limite = filas_limite;
	copia = LlenadoCeros(copia, fila_original, columna_original, filas_copia, columnas_copia, filas_limite, columnas_limite);
	copia = AjusteBordes(imagen, copia, fila_original, columna_original, filas_limite, columnas_limite);

	double** kernel = new double* [size_kernel];
	kernel = InicializaciónKernel(kernel, size_kernel, sigma, filas_limite, columnas_limite);

	Mat Gauss(fila_original, columna_original, CV_8UC1);
	Gauss = AplicaFiltroGauss(Gray, Gauss, kernel, size_kernel, fila_original, columna_original);

	/*Ecualizacion*/
	int v = 0;
	int y[256] = { 0 };
	int suma = 0;
	int limite = 0;
	double k = 255 / (Gauss.rows * Gauss.cols);
	
	Mat Ecualizada(fila_original, columna_original, CV_8UC1);
	for (int i = 0; i < Gauss.rows; i++) {
		for (int j = 0; j < Gauss.cols; j++) {
			v = Gauss.at<uchar>(Point(i, j));
			y[v] = y[v] + 1;
		}
	}

	for (int i = 0; i < Gauss.rows; i++) {
		for (int j = 0; j < Gauss.cols; j++) {
			limite = Gauss.at<uchar>(Point(i, j));
			for (int s = 0; s < limite; s++) {
				suma = suma + y[s];
			}
			/*En este caso ocupamos explicita el numero decimal ya que c++ no permite el uso de valores decimales muy pequeños */
			Ecualizada.at<uchar>(Point(i, j)) = 0.00064247921390778 * suma; /* k x suma */
			suma = 0;
		}
	}


	/*Sobel*/
	double** kernel_sobel_gx = new double* [3];

	kernel_sobel_gx[0] = new double[3];
	kernel_sobel_gx[1] = new double[3];
	kernel_sobel_gx[2] = new double[3];

	kernel_sobel_gx[0][0] = -1;
	kernel_sobel_gx[0][1] = 0;
	kernel_sobel_gx[0][2] = 1;

	kernel_sobel_gx[1][0] = -2;
	kernel_sobel_gx[1][1] = 0;
	kernel_sobel_gx[1][2] = 2;

	kernel_sobel_gx[2][0] = -1;
	kernel_sobel_gx[2][1] = 0;
	kernel_sobel_gx[2][2] = 1;


	Mat Ecualizada_expandida(fila_original + 2, columna_original + 2, CV_8UC1);

	/*Expandimos ecualizada para aplicar sobel*/
	for (int i = 0; i < Ecualizada_expandida.rows; i++) {
		for (int j = 0; j < Ecualizada_expandida.cols; j++) {
			if (i < 1 || i>fila_original || j < 1 || j>columna_original) {
				Ecualizada_expandida.at<uchar>(Point(i, j)) = 0;
			}
			else {
				Ecualizada_expandida.at<uchar>(Point(i, j)) = Ecualizada.at<uchar>(Point(i, j));
			}
		}
	}

	int i = 0;
	int j = 0;
	int m = 0;
	int n = 0;
	int pixel_f = 0;

	Mat Gx(fila_original, columna_original, CV_8UC1);
	for (i = 1; i < Ecualizada_expandida.rows-1; i++) {
		for (j = 1; j < Ecualizada_expandida.cols-1; j++) {
			pixel_f = Convolution(i, j, kernel_sobel_gx, Ecualizada_expandida);
			Gx.at<uchar>(Point(i-1, j-1)) = pixel_f;
		}
	}

	/*Expandimos Gx para aplicar Gy*/
	Mat Gx_Expandida(fila_original + 2, columna_original + 2, CV_8UC1);
	for (int i = 0; i < Gx_Expandida.rows; i++) {
		for (int j = 0; j < Gx_Expandida.cols; j++) {
			if (i < 1 || i>fila_original || j < 1 || j>columna_original) {
				Gx_Expandida.at<uchar>(Point(i, j)) = 0;
			}
			else {
				Gx_Expandida.at<uchar>(Point(i, j)) = Gx.at<uchar>(Point(i, j));
			}
		}
	}

	double** kernel_sobel_gy = new double* [3];

	kernel_sobel_gy[0] = new double[3];
	kernel_sobel_gy[1] = new double[3];
	kernel_sobel_gy[2] = new double[3];

	kernel_sobel_gy[0][0] = -1;
	kernel_sobel_gy[0][1] = -2;
	kernel_sobel_gy[0][2] = -1;

	kernel_sobel_gy[1][0] = 0;
	kernel_sobel_gy[1][1] = 0;
	kernel_sobel_gy[1][2] = 0;

	kernel_sobel_gy[2][0] = 1;
	kernel_sobel_gy[2][1] = 2;
	kernel_sobel_gy[2][2] = 1;

	Mat Gy(fila_original, columna_original, CV_8UC1);
	for (i = 1; i < Gx_Expandida.rows - 1; i++) {
		for (j = 1; j < Gx_Expandida.cols - 1; j++) {
			pixel_f = Convolution(i, j, kernel_sobel_gy, Gx_Expandida);
			Gy.at<uchar>(Point(i - 1, j - 1)) = pixel_f;
		}
	}


	pixel_f = 0;
	int pixel_gx = 0;
	int pixel_gy = 0;
	/*Definimos la matriz G absoluto*/
	Mat G_Absoluto(fila_original, columna_original, CV_8UC1);
	for (i = 0; i < imagen.rows; i++) {
		for (j = 0; j < imagen.cols; j++) {
			pixel_gx = Gx.at<uchar>(Point(i, j));
			pixel_gy = Gy.at<uchar>(Point(i, j));
			pixel_gx = pow(pixel_gx, 2);
			pixel_gy = pow(pixel_gy, 2);
			pixel_f = sqrt(pixel_gx+ pixel_gy);
			G_Absoluto.at<uchar>(Point(i, j )) = pixel_f;
		}
	}

	cout << "Kernel para el filtro de Gauss\n";
	cout << "--------------------------------------------------\n";
	for (i = 0; i < size_kernel;i++) {
		for (j = 0; j < size_kernel; j++) {
			cout << kernel[i][j]<<" ";
		}
		cout << "\n";
	}
	cout << "--------------------------------------------------\n";
	cout << "\n";

	namedWindow("Original", WINDOW_AUTOSIZE); 
	imshow("Original", imagen);

	namedWindow("Copia", WINDOW_AUTOSIZE);
	imshow("Copia", copia);

	namedWindow("Gauss", WINDOW_AUTOSIZE); 
	imshow("Gauss", Gauss);

	namedWindow("Ecualizada", WINDOW_AUTOSIZE); 
	imshow("Ecualizada", Ecualizada);

	namedWindow("Ecualizada_expandida", WINDOW_AUTOSIZE);
	imshow("Ecualizada_expandida", Ecualizada_expandida);

	namedWindow("G_Absoluto", WINDOW_AUTOSIZE);
	imshow("G_Absoluto", G_Absoluto);

	cout << "Filas Imagen Original: " << fila_original << endl;
	cout << "Columnas Imagen Original: " << columna_original << "\n\n";
	cout << "Filas Imagen Grises: " << Gray.rows << endl;
	cout << "Columnas Imagen Grises: " << Gray.cols << "\n\n";
	cout << "Filas Imagen con Borde: " << copia.rows << endl;
	cout << "Columnas Imagen con Borde: " << copia.cols << "\n\n";
	cout << "Filas Imagen con Filtro de Gauss: " << Gauss.rows << endl;
	cout << "Columnas Imagen con Filtro de Gauss: " << Gauss.cols << "\n\n";
	cout << "Filas Imagen Ecualizada: " << Ecualizada.rows << endl;
	cout << "Columnas Imagen Ecualizada: " << Ecualizada.cols << "\n\n";
	cout << "Filas Imagen G: " << Gy.rows << endl;
	cout << "Columnas Imagen G: " << Gy.cols << "\n\n";
	waitKey(0);
	return 1;
}