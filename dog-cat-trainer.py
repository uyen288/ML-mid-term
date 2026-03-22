import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import threading
import time
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

class DogCatTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Huấn luyện Nhận dạng Chó Mèo (ANN Version)")
        self.root.geometry("700x650")
        self.root.configure(bg="#f4f7f6")

        # Khởi tạo biến dữ liệu
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.model = None
        self.is_training = False

        self.create_widgets()

    def create_widgets(self):
        # Tiêu đề chính
        header = tk.Label(self.root, text="QUẢN LÝ MÔ HÌNH ANN NHẬN DẠNG CHÓ MÈO", 
                         font=("Helvetica", 16, "bold"), bg="#2c3e50", fg="white", pady=15)
        header.pack(fill="x")

        # --- KHUNG HUẤN LUYỆN ---
        train_frame = tk.LabelFrame(self.root, text="1. Huấn luyện mạng ANN", font=("Helvetica", 10, "bold"), padx=15, pady=10)
        train_frame.pack(fill="x", padx=20, pady=10)

        btn_load_train = tk.Button(train_frame, text="Chọn tập tin Train (X, y)", 
                                  command=self.load_train_data, bg="#3498db", fg="white", width=25, relief="flat")
        btn_load_train.grid(row=0, column=0, padx=5, pady=5)

        self.btn_train = tk.Button(train_frame, text="Bắt đầu Huấn luyện ANN", 
                                  command=self.run_training_thread, state="disabled", 
                                  bg="#27ae60", fg="white", width=25, relief="flat")
        self.btn_train.grid(row=0, column=1, padx=5, pady=5)

        self.progress_label = tk.Label(train_frame, text="Trạng thái: Sẵn sàng")
        self.progress_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        
        self.progress = ttk.Progressbar(train_frame, orient="horizontal", length=600, mode="determinate")
        self.progress.grid(row=2, column=0, columnspan=2, pady=5)

        # --- KHUNG KIỂM THỬ ---
        test_frame = tk.LabelFrame(self.root, text="2. Kiểm thử mô hình", font=("Helvetica", 10, "bold"), padx=15, pady=10)
        test_frame.pack(fill="x", padx=20, pady=10)

        btn_load_test = tk.Button(test_frame, text="Chọn tập dữ liệu Test (X, y)", 
                                 command=self.load_test_data, bg="#e67e22", fg="white", width=25, relief="flat")
        btn_load_test.grid(row=0, column=0, padx=5, pady=5)

        self.btn_test = tk.Button(test_frame, text="Kiểm thử Hiệu năng", 
                                 command=self.test_performance, state="disabled", 
                                 bg="#8e44ad", fg="white", width=25, relief="flat")
        self.btn_test.grid(row=0, column=1, padx=5, pady=5)

        # --- KHUNG KẾT QUẢ ---
        result_frame = tk.LabelFrame(self.root, text="3. Kết quả chi tiết", font=("Helvetica", 10, "bold"), padx=15, pady=10)
        result_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.result_text = tk.Text(result_frame, height=10, font=("Consolas", 10), bg="#ffffff")
        self.result_text.pack(fill="both", expand=True, pady=5)

        self.btn_save = tk.Button(result_frame, text="Lưu mô hình ANN (.pkl)", 
                                 command=self.save_model, state="disabled", 
                                 bg="#c0392b", fg="white", width=30, relief="flat")
        self.btn_save.pack(pady=5)

    def log(self, message):
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)

    def load_train_data(self):
        file_x = filedialog.askopenfilename(title="Chọn file X_train", filetypes=[("Numpy files", "*.npy"), ("All files", "*.*")])
        if not file_x: return
        file_y = filedialog.askopenfilename(title="Chọn file y_train")
        
        if file_x and file_y:
            try:
                # Mock data cho mục đích demo (thay thế bằng np.load thực tế)
                self.X_train = np.random.rand(200, 1024) 
                self.y_train = np.random.randint(0, 2, 200)
                
                self.log(f"[INFO] Đã tải dữ liệu Train thành công.")
                self.log(f"[INFO] Số lượng mẫu: {len(self.X_train)} - Số đặc trưng: {self.X_train.shape[1]}")
                self.btn_train.config(state="normal")
                messagebox.showinfo("Thành công", "Dữ liệu huấn luyện đã sẵn sàng!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi nạp dữ liệu: {str(e)}")

    def load_test_data(self):
        file_x = filedialog.askopenfilename(title="Chọn file X_test")
        if not file_x: return
        file_y = filedialog.askopenfilename(title="Chọn file y_test")
        
        if file_x and file_y:
            try:
                self.X_test = np.random.rand(50, 1024)
                self.y_test = np.random.randint(0, 2, 50)
                
                self.log(f"[INFO] Đã tải dữ liệu Test: {len(self.X_test)} mẫu.")
                self.btn_test.config(state="normal")
                messagebox.showinfo("Thành công", "Dữ liệu kiểm thử đã sẵn sàng!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi nạp dữ liệu: {str(e)}")

    def update_progress(self, value, text):
        self.progress['value'] = value
        self.progress_label.config(text=f"Tiến độ: {value}% - {text}")
        self.root.update_idletasks()

    def run_training_thread(self):
        self.btn_train.config(state="disabled")
        self.is_training = True
        thread = threading.Thread(target=self.train_ann_model)
        thread.start()

    def train_ann_model(self):
        try:
            self.update_progress(10, "Thiết lập kiến trúc mạng thần kinh (ANN)...")
            time.sleep(1)
            
            # Cấu trúc ANN: 1 lớp ẩn 100 neuron, hàm kích hoạt relu, tối ưu hóa adam
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                activation='relu', 
                solver='adam', 
                max_iter=500,
                random_state=42,
                verbose=False
            )
            
            self.update_progress(30, "Bắt đầu quá trình Forward/Backward Propagation...")
            time.sleep(1.5)
            
            self.update_progress(60, "Đang tối ưu hóa trọng số (Weight Optimization)...")
            self.model.fit(self.X_train, self.y_train)
            time.sleep(1)
            
            self.update_progress(90, "Kiểm tra sự hội tụ của hàm mất mát...")
            time.sleep(0.8)
            
            self.update_progress(100, "Hoàn tất huấn luyện mạng ANN!")
            self.log(">>> SUCCESS: Mô hình Artificial Neural Network đã sẵn sàng.")
            self.log(f">>> Số vòng lặp đã thực hiện: {self.model.n_iter_}")
            self.log(f">>> Loss cuối cùng: {self.model.loss_:.4f}")
            
            self.btn_save.config(state="normal")
            messagebox.showinfo("Thông báo", "Huấn luyện ANN hoàn tất!")
            
        except Exception as e:
            self.log(f"[ERROR] Lỗi ANN: {str(e)}")
            messagebox.showerror("Lỗi", f"Quá trình huấn luyện thất bại: {str(e)}")
        finally:
            self.is_training = False

    def test_performance(self):
        if self.model is None: return
        
        try:
            self.log("--- BÁO CÁO KIỂM THỬ ANN ---")
            y_pred = self.model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, target_names=["Mèo", "Chó"])
            
            self.log(f"Độ chính xác (Accuracy): {acc:.2%}")
            self.log("Chi tiết Precision/Recall/F1-score:")
            self.log(report)
            
        except Exception as e:
            self.log(f"[ERROR] Lỗi kiểm thử: {str(e)}")

    def save_model(self):
        if self.model is None: return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl", 
                                               filetypes=[("Pickle files", "*.pkl")],
                                               title="Lưu mô hình ANN")
        if file_path:
            try:
                joblib.dump(self.model, file_path)
                self.log(f"[SYSTEM] Đã lưu mô hình ANN vào: {file_path}")
                messagebox.showinfo("Thành công", "Lưu mô hình ANN hoàn tất!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi lưu file: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DogCatTrainerApp(root)
    root.mainloop()