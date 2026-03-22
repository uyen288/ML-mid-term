import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class DogCatPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận dạng Chó Mèo - Dự đoán & Phân loại Folder")
        self.root.geometry("1100x850")
        self.root.configure(bg="#ecf0f1")

        self.model = None
        self.image_path = None
        self.img_size = (32, 32) 
        self.thumbnails = [] # Lưu trữ references để tránh bị xóa khỏi bộ nhớ
        
        # Lưu trữ kết quả phân loại mới nhất để tính Confusion Matrix
        self.last_predictions = []
        self.last_filenames = []

        self.setup_ui()

    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="HỆ THỐNG NHẬN DẠNG & PHÂN LOẠI CHÓ MÈO", 
                         font=("Helvetica", 16, "bold"), bg="#2c3e50", fg="white", pady=15)
        header.pack(fill="x")

        # --- Khung Điều khiển ---
        control_frame = tk.Frame(self.root, bg="#ecf0f1", pady=10)
        control_frame.pack(fill="x", padx=20)

        self.btn_load_model = tk.Button(control_frame, text="1. Nạp Mô hình (.pkl)", 
                                        command=self.load_model, bg="#95a5a6", fg="white", width=20, relief="flat")
        self.btn_load_model.grid(row=0, column=0, padx=5, pady=5)

        self.btn_select_img = tk.Button(control_frame, text="2. Chọn 01 ảnh lẻ", 
                                       command=self.select_image, state="disabled",
                                       bg="#3498db", fg="white", width=20, relief="flat")
        self.btn_select_img.grid(row=0, column=1, padx=5, pady=5)

        self.btn_select_folder = tk.Button(control_frame, text="3. Phân loại Folder", 
                                          command=self.process_folder, state="disabled",
                                          bg="#8e44ad", fg="white", width=20, relief="flat")
        self.btn_select_folder.grid(row=0, column=2, padx=5, pady=5)

        # Nút xem Confusion Matrix
        self.btn_show_cm = tk.Button(control_frame, text="4. Xem Ma trận Nhầm lẫn", 
                                    command=self.show_confusion_matrix, state="disabled",
                                    bg="#e74c3c", fg="white", width=20, relief="flat")
        self.btn_show_cm.grid(row=0, column=3, padx=5, pady=5)

        # --- Khu vực Hiển thị chính ---
        main_content = tk.Frame(self.root, bg="#ecf0f1")
        main_content.pack(fill="both", expand=True, padx=20, pady=10)

        # Cột trái: Dự đoán ảnh đơn
        left_panel = tk.LabelFrame(main_content, text="Dự đoán ảnh đơn", bg="#ecf0f1", padx=10, pady=10)
        left_panel.pack(side="left", fill="y", padx=(0, 10))

        self.canvas = tk.Canvas(left_panel, width=250, height=250, bg="white", highlightthickness=1)
        self.canvas.pack(pady=10)
        self.img_item = self.canvas.create_image(125, 125, anchor=tk.CENTER)
        self.text_placeholder = self.canvas.create_text(125, 125, text="Chưa có ảnh", fill="gray")

        self.btn_predict = tk.Button(left_panel, text="DỰ ĐOÁN ẢNH NÀY", 
                                    command=self.predict, state="disabled",
                                    font=("Helvetica", 10, "bold"), bg="#27ae60", fg="white", width=20)
        self.btn_predict.pack(pady=5)

        self.lbl_result = tk.Label(left_panel, text="Kết quả: ...", font=("Helvetica", 12, "bold"), bg="#ecf0f1")
        self.lbl_result.pack(pady=5)

        # Cột phải: Kết quả Folder
        right_panel = tk.Frame(main_content, bg="#ecf0f1")
        right_panel.pack(side="right", fill="both", expand=True)

        self.create_scrollable_sections(right_panel)

    def create_scrollable_sections(self, parent):
        container = tk.Frame(parent, bg="#ecf0f1")
        container.pack(fill="both", expand=True)

        # Phần Chó
        dog_section = tk.LabelFrame(container, text="DANH SÁCH CHÓ (DOG)", fg="#2980b9", font=("bold", 10))
        dog_section.pack(side="left", fill="both", expand=True, padx=5)
        
        self.dog_canvas = tk.Canvas(dog_section, bg="white")
        self.dog_scrollbar = ttk.Scrollbar(dog_section, orient="vertical", command=self.dog_canvas.yview)
        self.dog_scroll_frame = tk.Frame(self.dog_canvas, bg="white")

        self.dog_scroll_frame.bind("<Configure>", lambda e: self.dog_canvas.configure(scrollregion=self.dog_canvas.bbox("all")))
        self.dog_canvas.create_window((0, 0), window=self.dog_scroll_frame, anchor="nw")
        self.dog_canvas.configure(yscrollcommand=self.dog_scrollbar.set)
        
        self.dog_canvas.pack(side="left", fill="both", expand=True)
        self.dog_scrollbar.pack(side="right", fill="y")

        # Phần Mèo
        cat_section = tk.LabelFrame(container, text="DANH SÁCH MÈO (CAT)", fg="#e67e22", font=("bold", 10))
        cat_section.pack(side="left", fill="both", expand=True, padx=5)

        self.cat_canvas = tk.Canvas(cat_section, bg="white")
        self.cat_scrollbar = ttk.Scrollbar(cat_section, orient="vertical", command=self.cat_canvas.yview)
        self.cat_scroll_frame = tk.Frame(self.cat_canvas, bg="white")

        self.cat_scroll_frame.bind("<Configure>", lambda e: self.cat_canvas.configure(scrollregion=self.cat_canvas.bbox("all")))
        self.cat_canvas.create_window((0, 0), window=self.cat_scroll_frame, anchor="nw")
        self.cat_canvas.configure(yscrollcommand=self.cat_scrollbar.set)

        self.cat_canvas.pack(side="left", fill="both", expand=True)
        self.cat_scrollbar.pack(side="right", fill="y")

        self.lbl_folder_stat = tk.Label(parent, text="Sẵn sàng phân loại", bg="#ecf0f1", font=("italic", 10))
        self.lbl_folder_stat.pack(pady=5)

    def load_model(self):
        file_path = filedialog.askopenfilename(title="Chọn mô hình đã train", filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            try:
                self.model = joblib.load(file_path)
                messagebox.showinfo("Thành công", "Đã nạp mô hình ANN!")
                self.btn_select_img.config(state="normal")
                self.btn_select_folder.config(state="normal")
                self.btn_show_cm.config(state="normal")
                self.btn_load_model.config(bg="#2ecc71")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể nạp mô hình: {e}")

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                self.image_path = file_path
                img = Image.open(file_path)
                img_display = img.resize((250, 250), Image.Resampling.LANCZOS)
                self.tk_img = ImageTk.PhotoImage(img_display)
                
                self.canvas.itemconfig(self.text_placeholder, state='hidden')
                self.canvas.itemconfig(self.img_item, image=self.tk_img, state='normal')
                
                self.btn_predict.config(state="normal")
                self.lbl_result.config(text="Sẵn sàng dự đoán", fg="black")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể hiển thị ảnh: {e}")

    def preprocess_image(self, path):
        img = Image.open(path).convert('L')
        img = img.resize(self.img_size)
        img_array = np.array(img).flatten()
        img_array = img_array / 255.0
        return img_array.reshape(1, -1)

    def predict(self):
        if not self.model or not self.image_path: return
        try:
            processed_img = self.preprocess_image(self.image_path)
            prediction = self.model.predict(processed_img)[0]
            label = "MÈO (CAT)" if prediction == 0 else "CHÓ (DOG)"
            color = "#e67e22" if prediction == 0 else "#2980b9"
            self.lbl_result.config(text=f"Kết quả: {label}", fg=color)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Dự đoán thất bại: {e}")

    def add_thumbnail_card(self, parent_frame, img_path, filename):
        card = tk.Frame(parent_frame, bg="white", pady=2)
        card.pack(fill="x", padx=2, pady=2)
        try:
            img = Image.open(img_path)
            img.thumbnail((40, 40))
            photo = ImageTk.PhotoImage(img)
            self.thumbnails.append(photo)
            lbl_img = tk.Label(card, image=photo, bg="white")
            lbl_img.pack(side="left", padx=5)
            lbl_name = tk.Label(card, text=filename, bg="white", font=("Arial", 8), anchor="w")
            lbl_name.pack(side="left", fill="x")
        except:
            pass

    def process_folder(self):
        folder_path = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
        if not folder_path: return

        for widget in self.dog_scroll_frame.winfo_children(): widget.destroy()
        for widget in self.cat_scroll_frame.winfo_children(): widget.destroy()
        self.thumbnails.clear()
        self.last_predictions = []
        self.last_filenames = []

        extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

        if not image_files:
            messagebox.showwarning("Thông báo", "Không tìm thấy file ảnh nào!")
            return

        dogs_count = 0
        cats_count = 0

        for filename in image_files:
            full_path = os.path.join(folder_path, filename)
            try:
                processed_img = self.preprocess_image(full_path)
                prediction = self.model.predict(processed_img)[0]
                
                self.last_predictions.append(prediction)
                self.last_filenames.append(filename)

                if prediction == 1: # Chó
                    self.add_thumbnail_card(self.dog_scroll_frame, full_path, filename)
                    dogs_count += 1
                else: # Mèo
                    self.add_thumbnail_card(self.cat_scroll_frame, full_path, filename)
                    cats_count += 1
            except:
                continue

        self.lbl_folder_stat.config(text=f"Hoàn tất! Tìm thấy {dogs_count} Chó và {cats_count} Mèo")
        messagebox.showinfo("Xong", "Phân loại thư mục hoàn tất!")

    def show_confusion_matrix(self):
        """
        Hiển thị cửa sổ mới chứa Ma trận nhầm lẫn.
        """
        cm_window = tk.Toplevel(self.root)
        cm_window.title("Confusion Matrix - Ma trận nhầm lẫn")
        cm_window.geometry("700x700")

        # Logic: Giả định nhãn thực tế dựa trên tên file
        y_true = []
        y_pred = self.last_predictions
        
        valid_data = False
        if len(y_pred) > 0:
            for fname in self.last_filenames:
                name_lower = fname.lower()
                # Kiểm tra nếu tên file có chứa từ 'dog' hoặc 'cat'
                if 'dog' in name_lower: y_true.append(1)
                elif 'cat' in name_lower: y_true.append(0)
                else: y_true.append(-1) 
            
            # Lọc ra những file xác định được nhãn thực tế
            indices = [i for i, val in enumerate(y_true) if val != -1]
            y_true_final = [y_true[i] for i in indices]
            y_pred_final = [y_pred[i] for i in indices]
            
            if len(y_true_final) > 0:
                cm = confusion_matrix(y_true_final, y_pred_final, labels=[0, 1])
                valid_data = True
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        if valid_data:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
            disp.plot(ax=ax, cmap='Blues', values_format='d')
            ax.set_title(f"Ma trận dựa trên {len(y_true_final)} ảnh có nhãn trong tên file")
        else:
            # Hiển thị thông báo khi không có dữ liệu đối chứng
            ax.text(0.5, 0.5, "KHÔNG CÓ DỮ LIỆU ĐỐI CHỨNG\n\nĐể xem ma trận nhầm lẫn, bạn hãy đặt tên file ảnh\ncó chứa từ 'dog' hoặc 'cat'.\nVí dụ: dog_01.jpg, my_cat.png", 
                    ha='center', va='center', fontsize=10, color='red', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Ma trận nhầm lẫn (Chưa có dữ liệu)")

        canvas = FigureCanvasTkAgg(fig, master=cm_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = DogCatPredictorApp(root)
    root.mainloop()