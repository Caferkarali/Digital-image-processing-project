# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:10:18 2024

@author: Cafer karali
"""
import cv2
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
import tkinter as tk 
resim = None
etiket = None
rotation_angle = 0



def dosyaAc():
    global etiket, resim, original_image
    resim_yolu = filedialog.askopenfilename(initialdir="C:/Users/Cafer Karali/Desktop", title="Bir dosya seç...")
    print(resim_yolu)

    original_image = cv2.imread(resim_yolu)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # OpenCV, BGR formatında okur, bu yüzden RGB'ye dönüştürüyoruz

    img = Image.open(resim_yolu)
    img = ImageTk.PhotoImage(img)

    etiket = Label(pencere, image=img)
    etiket.image = img
    etiket.grid(row=1, column=1, padx=20, pady=20)



def select_image_tool():
    dosyaAc()


def menu1():
    menu1 = Toplevel(pencere)
    submenu = Menu(menu1)
    submenu.add_command(label="Dosya Aç", command=dosyaAc)
    submenu.add_command(label="Histogram Oluştur", command=histogram_olustur)
    submenu.add_command(label="Ana Menüye Dön", command=menu1.destroy)
    menu1.config(menu=submenu)


def odev1_action():
    print("Ödev 1: Temel İşlevselliği Oluştur")
    menu1()


def menu2():
    menu2 = Toplevel(pencere)
    submen = Menu(menu2)
    submen.add_command(label="Dosya Aç", command=dosyaAc)
    submen.add_command(label="Thresholding yap", command=simple_thresholding)
    submen.add_command(label="Ana Menüye Dön", command=menu2.destroy)
    menu2.config(menu=submen)


def menu3():
    menu3 = Toplevel(pencere)
    subme = Menu(menu3)
    subme.add_command(label="Dosya Aç", command=dosyaAc)
    subme.add_command(label="flip yap", command=flip)
    subme.add_command(label="Ana Menüye Dön", command=menu3.destroy)
    menu3.config(menu=subme)


def menu4():
    menu4 = Toplevel(pencere)
    subm = Menu(menu4)
    subm.add_command(label="Dosya Aç", command=dosyaAc)
    subm.add_command(label="Görüntü Boyutunu Büyütme", command=goruntuBuyut)
    subm.add_command(label="Görüntü Boyutunu Küçültme", command=goruntuKucult)
    subm.add_command(label="Zoom In ve Zoom Out", command=goruntuZoom)
    subm.add_command(label="Görüntü Döndürme", command=goruntuDondur_gui)
    subm.add_command(label="Ana Menüye Dön", command=menu4.destroy)
    menu4.config(menu=subm)


def menu5():
    menu5 = Toplevel(pencere)
    sub = Menu(menu5)
    sub.add_command(label="Dosya Aç", command=dosyaAc)
    sub.add_command(label="Soru 1:Sigmoid Fonksiyonları ve  S- Curve metodu ile kontrast artırma ",
                    command=kontrast_artırma)
    sub.add_command(label="Soru 2:Hough Transform ile yol tespiti ve göz tespiti", command=Hough_Transform)
    sub.add_command(label="Soru 3:Deblurring algoritması geliştirme", command=deblurring)
    sub.add_command(label="Soru 4:Resimdeki nesneleri sayma ve özellik çıkarma", command=Ozellik_cikarma)
    sub.add_command(label="Ana Menüye Dön", command=menu5.destroy)
    menu5.config(menu=sub)

def Hough_Transform():
    def detect_lines():
        # Resmi oku
        img_path = filedialog.askopenfilename()
        if not img_path:
            return

        global img  # Define img as global to make it accessible outside this function
        img = cv2.imread(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 75, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, maxLineGap=250)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.imshow("Edges", edges)
        cv2.imshow("Image", img)
        cv2.imwrite("1.3_lines_with_gap.jpg", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_eyes():
        image_path = filedialog.askopenfilename()
        if not image_path:
            return

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 150)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=30, minRadius=10,
                                   maxRadius=40)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        cv2.imshow('Detected Eyes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def main():
        root = Tk()
        root.title("Image Processing")

        button_frame = Frame(root)
        button_frame.pack(padx=10, pady=10)

        lines_button = Button(button_frame, text="Çizgileri Bul", command=detect_lines)
        lines_button.grid(row=0, column=0, padx=5, pady=5)

        eyes_button = Button(button_frame, text="Gözleri Bul", command=detect_eyes)
        eyes_button.grid(row=0, column=1, padx=5, pady=5)

        root.mainloop()

    if __name__ == "__main__":
        main()

def Ozellik_cikarma():


    # Görüntü yolunu belirtin
    goruntu_yolu = r'C:\\Users\\Cafer Karali\\Downloads\\say.jpg'

    # Görüntüyü yükleyin
    hiperspektral_goruntu = cv2.imread(goruntu_yolu)

    # Görüntüyü HSV renk uzayına dönüştürün
    hsv_goruntu = cv2.cvtColor(hiperspektral_goruntu, cv2.COLOR_BGR2HSV)

    # Yeşil tonlarının HSV aralığını belirleyin
    alt_yesil = np.array([35, 50, 50])  # Düşük yeşil tonları
    ust_yesil = np.array([80, 255, 255])  # Yüksek yeşil tonları

    # Yeşil tonlarını içeren maske oluşturun
    yesil_maske = cv2.inRange(hsv_goruntu, alt_yesil, ust_yesil)

    # Koyu yeşil bölgelerin konumlarını (koordinatlarını) alın
    koyu_yesil_konumlar = np.argwhere(yesil_maske > 0)

    # Özellikler için boş listeler oluşturun
    ozellikler = []

    # Her bir nokta için özellikleri hesaplayın
    for konum in koyu_yesil_konumlar:
        x, y = konum
        center = np.sqrt((x - hiperspektral_goruntu.shape[0] / 2) ** 2 + (y - hiperspektral_goruntu.shape[1] / 2) ** 2)
        length = x
        width = y
        diagonal = np.sqrt(x ** 2 + y ** 2)
        energy = x ** 2 + y ** 2
        entropy_value = entropy([x, y], base=2)
        mean = (x + y) / 2
        median = np.median([x, y])

        # Her bir özelliği ilgili listeye ekleyin
        ozellikler.append([center, length, width, diagonal, energy, entropy_value, mean, median])

    # DataFrame oluşturun
    df = pd.DataFrame(ozellikler,
                      columns=["Center", "Length", "Width", "Diagonal", "Energy", "Entropy", "Mean", "Median"])

    # Dosya yolunu belirtin
    dosya_yolu = r'C:\\Users\\Cafer Karali\\Downloads\\hiperspektral_ozellikler.xlsx'

    # DataFrame'i Excel dosyasına kaydedin
    df.to_excel(dosya_yolu, index=False)

    print(f"Hiperspektral özellikler başarıyla '{dosya_yolu}' dosyasına kaydedildi.")


def deblurring():
    global original_image
    restored_image = cv2.medianBlur(original_image, 5)  # Görüntüyü işleme

    # 3. Kenar algılama
    edges = cv2.Canny(restored_image, 100, 200)

    # 4. Morfolojik işlemler
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)

    # 5. Bölümleme
    contours, _ = cv2.findContours(eroded_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(original_image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Motion blur düzeltme
    for i in range(3):  # Red, Green, Blue kanalları
        restored_image[:, :, i] = cv2.inpaint(restored_image[:, :, i], eroded_edges, inpaintRadius=3,
                                              flags=cv2.INPAINT_TELEA)

    # Sonuçları görselleştirme
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Restored Image', restored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def kontrast_artırma():
    def standard_sigmoid(x, a=1):
        return 1 / (1 + np.exp(-a * x))

    def shifted_sigmoid(x, a=1, b=0):
        return 1 / (1 + np.exp(-a * (x - b)))

    def tilted_sigmoid(x, a=1, b=0):
        return 1 / (1 + np.exp(-a * (x - b))) - 0.5

    # Kendi ürettiğiniz fonksiyon
    def karisik_sigmoid(x, a1=1, a2=1, a3=1, b1=0, b2=0, b3=0, c1=1, c2=1, c3=1):
        standard = standard_sigmoid(x, a=a1) * c1
        shifted = shifted_sigmoid(x, a=a2, b=b2) * c2
        tilted = tilted_sigmoid(x, a=a3, b=b3) * c3
        return standard + shifted + tilted

    # S-eğrisi metodu ile kontrast güçlendirme
    def s_curve_contrast(image, sigmoid_func, **kwargs):
        # Görüntü değerlerini [0, 1] aralığına normalize et
        normalized_image = image.astype(float) / 255.0

        # Sigmoid fonksiyonunu uygula
        transformed_image = sigmoid_func(normalized_image, **kwargs)

        # [0, 1] aralığını [0, 255] aralığına dönüştür
        transformed_image = np.round(transformed_image * 255).astype(np.uint8)

        return transformed_image

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    img_path = filedialog.askopenfilename()  # Ask the user to select a file

    if not img_path:
        print("No image selected.")
        return

    # Load the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load image.")
        return

    goruntu_standard = s_curve_contrast(image, standard_sigmoid)
    goruntu_shifted = s_curve_contrast(image, shifted_sigmoid)
    goruntu_tilted = s_curve_contrast(image, tilted_sigmoid)
    goruntu_karisik = s_curve_contrast(image, karisik_sigmoid)

    cv2.imshow("Orjinal Görüntü", image)
    cv2.imshow("Standard Sigmoid ", goruntu_standard)
    cv2.imshow("Shifted Sigmoid ", goruntu_shifted)
    cv2.imshow("Tilted Sigmoid ", goruntu_tilted)
    cv2.imshow("Karisik Sigmoid ", goruntu_karisik)

    # Bekle ve pencereyi kapat
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function to start the process


def odev2_action():
    menu2()


def histogram_olustur():
    img = Image.open(resim).convert('L')
    histogram = img.histogram()
    histogram_pencere = Toplevel()
    histogram_pencere.title('Histogram')
    plt.plot(histogram, color='black')
    plt.xlabel('Pixel Değerleri')
    plt.ylabel('Frekans')
    plt.title('Histogram')
    canvas = FigureCanvasTkAgg(plt.gcf(), master=histogram_pencere)
    canvas.draw()
    canvas.get_tk_widget().pack()
    histogram_pencere.mainloop()


def simple_thresholding():
    global resim
    img = Image.open(resim).convert('L')
    thresholded_img = img.point(lambda p: p > 128 and 255)
    img.show()
    thresholded_img.show()


def goruntuBuyut():
    global resim, original_width, original_height
    if resim is not None:
        root = Tk()
        root.title("Görüntü Boyutlandırma")

        new_width_value = IntVar()
        new_height_value = IntVar()

        def set_new_width():
            width_window = Toplevel(root)
            width_window.title('Yeni Genişlik Ayarı')
            width_scale = Scale(width_window, from_=0, to=1000, orient=HORIZONTAL, label="Yeni Genişlik",
                                variable=new_width_value)
            width_scale.pack(padx=10, pady=10)

        def set_new_height():
            height_window = Toplevel(root)
            height_window.title('Yeni Yükseklik Ayarı')
            height_scale = Scale(height_window, from_=0, to=1000, orient=HORIZONTAL, label="Yeni Yükseklik",
                                 variable=new_height_value)
            height_scale.pack(padx=10, pady=10)

        def bilinear_interpolate(image, x, y):
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, original_width - 1), min(y1 + 1, original_height - 1)

            Q11 = image[y1, x1]
            Q12 = image[y1, x2]
            Q21 = image[y2, x1]
            Q22 = image[y2, x2]

            R1 = ((x2 - x) * Q11 + (x - x1) * Q12)
            R2 = ((x2 - x) * Q21 + (x - x1) * Q22)

            return ((y2 - y) * R1 + (y - y1) * R2) / ((x2 - x1) * (y2 - y1))

        def resize_image_bilinear(image, new_width, new_height):
            resized_image = np.zeros((new_height, new_width), dtype=np.uint8)

            x_ratio = (original_width - 1) / (new_width - 1)
            y_ratio = (original_height - 1) / (new_height - 1)

            for y in range(new_height):
                for x in range(new_width):
                    original_x = x * x_ratio
                    original_y = y * y_ratio

                    x1, y1 = int(original_x), int(original_y)
                    x2, y2 = min(x1 + 1, original_width - 1), min(y1 + 1, original_height - 1)

                    weight_x2_x = original_x - x1
                    weight_x_x1 = x2 - original_x
                    weight_y2_y = original_y - y1
                    weight_y_y1 = y2 - original_y

                    interpolated_value = (image[y1, x1] * weight_x_x1 * weight_y_y1 +
                                          image[y1, x2] * weight_x2_x * weight_y_y1 +
                                          image[y2, x1] * weight_x_x1 * weight_y2_y +
                                          image[y2, x2] * weight_x2_x * weight_y2_y)
                    resized_image[y, x] = interpolated_value

            return resized_image

        def resize_and_show():
            new_width = new_width_value.get()
            new_height = new_height_value.get()

            if resim:
                image = Image.open(resim).convert('L')
                global original_width, original_height
                original_width, original_height = image.size

                resized_image = resize_image_bilinear(np.array(image), new_width, new_height)

                resized_image_pil = Image.fromarray(resized_image)

                tk_image = ImageTk.PhotoImage(resized_image_pil)

                image_label = Label(root, image=tk_image)
                image_label.image = tk_image
                image_label.pack()

                root.mainloop()
            else:
                print("Lütfen önce bir resim seçin.")

        btn_width = Button(root, text="Yeni Genişlik Ayarı", command=set_new_width)
        btn_width.pack(padx=10, pady=10)

        btn_height = Button(root, text="Yeni Yükseklik Ayarı", command=set_new_height)
        btn_height.pack(padx=10, pady=10)

        btn_resize = Button(root, text="Boyutlandır ve Göster", command=resize_and_show)
        btn_resize.pack(padx=10, pady=10)

        root.mainloop()
    else:
        print("Lütfen önce bir resim seçin.")


def goruntuKucult():
    def bicubic_interpolate(img, new_width, new_height):
        height, width, channels = img.shape

        new_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)

        x_ratio = width / new_width
        y_ratio = height / new_height

        for channel in range(channels):
            for i in range(new_height):
                for j in range(new_width):
                    x, y = (j * x_ratio, i * y_ratio)

                    x_floor, y_floor = np.floor([x, y]).astype(int)
                    x_ceil, y_ceil = np.ceil([x, y]).astype(int)

                    if x_ceil >= width:
                        x_ceil = x_floor
                    if y_ceil >= height:
                        y_ceil = y_floor

                    bl = img[y_floor, x_floor, channel]
                    br = img[y_floor, x_ceil, channel]
                    tl = img[y_ceil, x_floor, channel]
                    tr = img[y_ceil, x_ceil, channel]

                    a = x - x_floor
                    b = y - y_floor

                    pixel_value = bl * (1 - a) * (1 - b) + br * a * (1 - b) + tl * (1 - a) * b + tr * a * b

                    new_img[i, j, channel] = pixel_value

        return new_img

    resize_ratio_label = Label(pencere, text="Yeniden Boyutlandırma Oranı:")
    resize_ratio_label.grid(row=2, column=0, padx=10, pady=10)

    resize_ratio_entry = Entry(pencere)
    resize_ratio_entry.pack()

    img = plt.imread(resim)

    new_width = int(img.shape[1] * resize_ratio)
    new_height = int(img.shape[0] * resize_ratio)
    resized_img = bicubic_interpolate(img, new_width, new_height)

    plt.subplot(121), plt.imshow(img), plt.title('Original ')
    plt.subplot(122), plt.imshow(resized_img), plt.title('küçültülmüş resim')
    plt.show()


def rotate_image_nearest_neighbor(image, angle):
    return image.rotate(angle, resample=Image.NEAREST)


def goruntuDondur():
    global resim, rotation_angle
    if resim is not None:
        img = Image.open(resim)
        rotated_img = rotate_image_nearest_neighbor(img, rotation_angle)
        rotated_img.show()
    else:
        print("Lütfen önce bir resim seçin.")


def goruntuDondur_gui():
    global rotation_angle
    rotate_window = Toplevel(pencere)
    rotate_window.title('Görüntü Döndürme Açısı')
    Label(rotate_window, text="Döndürme Açısı (derece):").grid(row=0, column=0)
    entry_angle = Entry(rotate_window)
    entry_angle.grid(row=0, column=1)
    Button(rotate_window, text="Döndür", command=lambda: rotate_image_action(entry_angle.get())).grid(row=1, column=0,
                                                                                                      columnspan=2)


def rotate_image_action(angle_str):
    global rotation_angle
    try:
        rotation_angle = float(angle_str)
        goruntuDondur()
    except ValueError:
        print("Geçersiz döndürme açısı.")


def custom_interpolation(im, zoom_factor):
    height, width, channels = im.shape
    new_height = int(height * zoom_factor)
    new_width = int(width * zoom_factor)

    y_indices = np.arange(new_height).repeat(new_width)
    x_indices = np.tile(np.arange(new_width), new_height)

    y_original = np.clip((y_indices / zoom_factor).astype(int), 0, height - 1)
    x_original = np.clip((x_indices / zoom_factor).astype(int), 0, width - 1)

    enlarged_img = im[y_original, x_original]

    return enlarged_img.reshape(new_height, new_width, channels)


def goruntuZoom():
    def zoom_image():
        try:
            zoom_factor = float(entry_zoom.get())
            if resim:
                im = np.array(Image.open(resim).convert("RGB"))
                enlarged_img = custom_interpolation(im, zoom_factor)
                enlarged_img_pil = Image.fromarray(enlarged_img, 'RGB')
                enlarged_img_pil.show()
            else:
                print("Lütfen önce bir resim seçin.")
        except ValueError:
            print("Geçersiz zoom faktörü!")

    rotate_window = Toplevel(pencere)
    rotate_window.title('Zooming')

    Label(rotate_window, text="zoom faktörünü girin: ").grid(row=0, column=0)
    entry_zoom = Entry(rotate_window)
    entry_zoom.grid(row=0, column=1)
    Button(rotate_window, text="Zoom", command=zoom_image).grid(row=1, column=0, columnspan=2)

    pencere.wait_window(rotate_window)


def odev3_action():
    menu3()


def odev4_action():
    menu4()


def vize_action():
    menu5()


def flip():
    img = Image.open(resim)
    grayscale_img = img.convert('L')
    flipped_img = grayscale_img.transpose(Image.FLIP_TOP_BOTTOM)
    grayscale_img.show()
    flipped_img.show()


pencere = Tk()
pencere.title("Görüntü işleme alanı")
pencere.config(bg="skyblue")

title_label = Label(pencere, text="Ders Adı: Dijital Görüntü İşleme    Öğrenci No: 211229027 Adı Soyadı: Cafer Karalı",
                    font=('Arial', 12), bg='skyblue')
title_label.grid(row=0, column=0, columnspan=2, pady=10)

left_frame = Frame(pencere, width=200, height=400, bg='grey')
left_frame.grid(row=1, column=0, padx=10, pady=5)

tool_bar = Frame(left_frame, width=400, height=900)
tool_bar.grid(row=2, column=0, padx=5, pady=5)

Label(tool_bar, text="Menü", relief=RAISED).grid(row=0, column=0, padx=5, pady=3, ipadx=10)

Button(tool_bar, text="Ödev 1:Temel İşlevselliği Oluştur (ödev1 örnek)", command=odev1_action).grid(row=2, column=0,
                                                                                                    padx=5, pady=5)
Button(tool_bar, text="Ödev 2:Filtre Uygulama(ödev1 örnek)", command=odev2_action).grid(row=3, column=0, padx=5, pady=5)
Button(tool_bar, text="Ödev 3(ödev1 örnek)", command=odev3_action).grid(row=4, column=0, padx=5, pady=5)
Button(tool_bar, text="Görüntü İşleme Ödevi 2: Temel Görüntü Operasyonları ve İnterpolasyon",
       command=odev4_action).grid(row=5, column=0, padx=5, pady=5)
Button(tool_bar, text="Dijital Görüntü İşleme Dersi Vize Ödevi", command=vize_action).grid(row=6, column=0, padx=5,
                                                                                           pady=5)
buton = Button(tool_bar, text="Dosya Seç", command=select_image_tool)
buton.grid(row=1, column=0, padx=5, pady=5)

pencere.mainloop()
