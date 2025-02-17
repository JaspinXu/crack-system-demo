import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import tkinter.font as tkFont

import pandas as pd
from PIL import Image, ImageTk
import json
from train.train import UNetModel
from detect import test
from estimate import est

class WinGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.__win()
        self.tk_list_box_m2dbzpwc = self.__tk_list_box_m2dbzpwc(self)
        self.tk_label_image = self.__tk_label_image(self)
        self.tk_label_m2dcq27y = self.__tk_label_m2dcq27y(self)
        self.tk_label_m2ddc36o = self.__tk_list_box_m2ddc36o(self)
        self.tk_list_box_m2ddc36o = self.__tk_list_box_m2ddc36o(self)
        self.show_rectangle()  # 在这里调用 show_rectangle 方法
        self.current_path = ''
        self.current_image = ''
        # 读取配置文件
        self.config_file = "configs\config.json"
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                config = json.load(f)
                if "last_folder" in config:
                    self.image_folder = config["last_folder"]
                if "output_folder" in config:
                    self.output_folder = config["output_folder"]
                    self.tk_label_output.config(text=f"输出位置：{os.path.basename(self.output_folder)}\\")
                if "current_image" in config:
                    self.current_image = config["current_image"]
                else:
                    self.tk_label_output.config(text=f"未选择输出位置")
                self.update_images()

    def save_config(self):
        config = {"last_folder": self.image_folder, "output_folder": self.output_folder,
            "current_image": self.current_image}
        with open(self.config_file, "w") as f:
            json.dump(config, f)

    def __win(self):
        self.title("山东大学裂缝检测与健康评估系统")
        # 设置窗口大小、居中
        width = 1280
        height = 720
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)

        # 创建画布并设置背景图片
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.canvas.pack(fill="both", expand=True)
        self.bg_image = Image.open("configs/background.jpg").resize((width, height), Image.Resampling.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        # 添加按钮和部件到画布上
        btn_open_folder = tk.Button(self, text="导入数据集", command=self.select_folder, bg="#1b69b4", fg="white", bd=0)
        btn_open_folder.place(x=973, y=28, width=73, height=14)
        # 在x=1072, y=27的位置增加一个宽77，高16的按钮
        btn_crack_detection = tk.Button(self, text="裂缝提取", command=self.crack_detect, bg="#1f5582",fg="white", bd=0)
        btn_crack_detection.place(x=1077, y=28, width=67, height=14)
        # 在x=1172, y=27的位置增加一个宽77，高16的按钮
        btn_start_calculation = tk.Button(self, text="开始计算", command=self.start_calculation, bg="#1f5582",fg="white", bd=0)
        btn_start_calculation.place(x=1177, y=28, width=67, height=14)

        # 添加一个标签
        self.tk_label_output = tk.Label(self, bg="#041022", foreground="white", font=("microsoft yahei", 12), bd=0, anchor="w")
        self.tk_label_output.place(x=936, y=60, width=301, height=33)
        self.tk_label_output.config(text="未选择输出位置")

    def scrollbar_autohide(self, vbar, hbar, widget):
        """自动隐藏滚动条"""
        def show():
            if vbar: vbar.lift(widget)
            if hbar: hbar.lift(widget)

        def hide():
            if vbar: vbar.lower(widget)
            if hbar: hbar.lower(widget)

        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Leave>", lambda e: hide())
        if hbar: hbar.bind("<Enter>", lambda e: show())
        if hbar: hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())

    def v_scrollbar(self, vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')

    def h_scrollbar(self, hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')

    def create_bar(self, master, widget, is_vbar, is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = tk.Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = tk.Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)

    def select_folder(self):
        self.image_folder = filedialog.askdirectory(title="选择数据集")
        self.output_folder = ''
        self.update_images()
        self.save_config()  # 保存配置文件

    def crack_detect(self):
        self.output_folder = filedialog.askdirectory(title="选择输出文件夹")
        if self.output_folder:
            self.tk_label_output.config(text=f"提取中...")
            self.save_config()  # 保存配置文件
            test(UNetModel().cuda(), self.image_folder, self.output_folder)
            self.tk_label_output.config(text=f"提取完成。点击图片查看结果")
            tk.messagebox.showwarning("提示", "提取完成！")
        else:
            self.tk_label_output.config(text=f"未选择输出位置")

    def start_calculation(self):
        self.tk_label_output.config(text=f"计算中...")
        # read_and_process_images(self.image_folder, self.output_folder + "\\metrics_samples.csv")
        est(self.image_folder)
        self.calculate_value(self.current_image)
        self.tk_label_output.config(text=f"计算完成")
        tk.messagebox.showwarning("提示", "计算完成！")

    def update_images(self):
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))]
        self.tk_list_box_m2dbzpwc.delete(0, tk.END)
        for image_file in image_files:
            self.tk_list_box_m2dbzpwc.insert(tk.END, image_file)
        if image_files:
            self.tk_list_box_m2dbzpwc.selection_set(0)
            self.update_image()
        else:
            self.show_rectangle()

    def __tk_list_box_m2dbzpwc(self, parent):
        font = tkFont.Font(family="microsoft yahei", size=18)  # 设置字体大小为18，加粗
        lb = tk.Listbox(parent, font=font, bg="#081c31", selectbackground="#113e60", fg="white",
                        highlightbackground="#081c31", bd=0)  # 设置背景颜色、选中后的背景颜色、字体颜色和边框颜色
        lb.bind("<<ListboxSelect>>", self.update_image)
        lb.place(x=89, y=146, width=265, height=336)
        self.create_bar(parent, lb, True, False, 59, 146, 295, 336, 1280, 720)
        return lb

    def __tk_label_image(self, parent):
        label = tk.Label(parent, image=self.bg_photo)
        label.place(x=458, y=131, width=369, height=366)  # 修改位置和大小
        label.bind("<Button-1>", self.toggle_image)  # 绑定左键单击事件到toggle_image方法
        return label

    def __tk_label_m2dcq27y(self, parent, text="———", fg="white"):
        font = tkFont.Font(family="microsoft yahei", size=25, weight="bold")  # 设置字体大小为30，加粗
        label = tk.Label(parent, font=font, bg="#081c31", fg=fg, anchor="center")  # 设置文本居中
        label.place(x=1130, y=469, width=70, height=48)
        label.config(text= text)  # 在标签中显示数字100
        return label

    def __tk_list_box_m2ddc36o(self, parent):
        font = tkFont.Font(family="microsoft yahei", size=17)  # 设置字体大小为18，加粗
        lb = tk.Listbox(parent, font=font, bg="#041022", selectbackground="#041022", fg="white",
                        highlightbackground="#041022", bd=0, selectmode=tk.BROWSE)  # 设置背景颜色、选中后的背景颜色、字体颜色和边框颜色
        lb.place(x=936, y=136, width=260, height=282)
        return lb

    def update_image(self, event=None):
        selected_image = self.current_image
        self.current_path = self.output_folder
        self.toggle_image()

        self.calculate_value(selected_image)

    def calculate_value(self, image_name):
        self.tk_list_box_m2ddc36o.delete(0, tk.END)
        # 读取CSV文件并获取数据
        if self.output_folder:
            flag = os.path.exists(self.output_folder + "\\result_scores.csv")
        else:
            flag = False
            self.__tk_label_m2dcq27y(self, text="———", fg="white")
        if flag:
            df = pd.read_csv(self.output_folder + "\\result_scores.csv")
            first_column_ = list(df[image_name])
            # print(first_column)
            score = first_column_[0]
            # print(score)
            first_column = first_column_[1:]
            # print(first_column[1:])
            # 保留每个数值的2位小数
            # first_column = [round(x, 2) for x in first_column]
            self.__tk_label_m2dcq27y(self, text=round(float(score),1), fg="red")
        else:
            first_column = ["———" for _ in range(20)]
        name = ["裂纹长度：", "裂纹面积：", "最大裂纹宽度：", "平均裂纹宽度：", "裂纹密度：", "标称平均宽度：", "分形维度：",
                "裂纹间距：", "最大裂纹高度：", "平均裂纹高度：", "平均裂纹形状系数：", "平均裂纹取向：", "裂纹体积：",
                "裂纹扩展速率：", "裂纹宽度变化率：", "裂纹分布均匀性：",
                "裂纹分布方向性：","裂缝分布集中度：", "裂缝分布离散度：", "可微调基础指标："]
        # "裂纹长度：",

        # 将第一列数据添加到列表框中
        self.tk_list_box_m2ddc36o.delete(0, tk.END)
        for index, item in enumerate(first_column):
            value_with_prefix = f"{name[index]}{item}"
            self.tk_list_box_m2ddc36o.insert(tk.END, value_with_prefix)

    def toggle_image(self, event=None):
        if self.tk_list_box_m2dbzpwc.curselection():
            selected_image = self.tk_list_box_m2dbzpwc.get(self.tk_list_box_m2dbzpwc.curselection())
            self.current_image = selected_image
            self.save_config()
        else:
            selected_image = self.current_image
        if self.current_path == self.image_folder:
            self.current_path = self.output_folder
        elif self.current_path ==self.output_folder:
            self.current_path = self.image_folder
        image_path = os.path.join(self.current_path, selected_image)
        if os.path.isfile(image_path):
            image = Image.open(image_path).resize((369, 366), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.tk_label_image.config(image=photo)
            self.tk_label_image.image = photo
        else:
            tk.messagebox.showwarning("警告", "请先提取裂缝！")


    def show_rectangle(self):
        white_image = Image.new('RGB', (369, 366), color='#041022')
        photo = ImageTk.PhotoImage(white_image)
        self.tk_label_image.config(image=photo)
        self.tk_label_image.image = photo

if __name__ == "__main__":
    win = WinGUI()
    win.minsize(width=1280, height=720)
    win.maxsize(width=1280, height=720)
    win.mainloop()
