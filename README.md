# Build-GUI-Python-for-Multidimensional-Scaling
this is the code to create a  Graphical User Interface for multidimensional scaling metrics using idle (python)

try:

    import tkinter as tk
    import tkinter.filedialog
    from tkinter.filedialog import askopenfilename
    import numpy as np
    import warnings
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import euclidean_distances
    
except ImportError:

    import tkinter as tk
    import tkinter.filedialog
    from tkinter.filedialog import askopenfilename
    import numpy as np
    import warnings
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import euclidean_distances


import pandas as pd

from PIL import Image, ImageTk


class MyWindow:

    def __init__(self, parent):
        self.parent=parent
        self.filename = None
        self.filename1 = None
        self.filename2 = None
        self.df = None
        self.df1 = None
        self.df2 = None
        self.parent = parent
        self.gambar1 = Image.open(r'd:/Picture4.jpg')
        self.gambar = ImageTk.PhotoImage(self.gambar1)
        self.label = tk.Label(parent, image=self.gambar)
        self.label.pack()
        self.button = tk.Button(root, text='Load Data',bg='white', command=self.load).place(x=30,y=10,height=30)
        self.button = tk.Button(root, text='Load Nama Objek',bg='white', command=self.load1).place(x=30,y=50,height=30)
        self.button = tk.Button(root, text='Load Nama Variabel',bg='white', command=self.load2).place(x=30,y=90,height=30)
        self.button = tk.Button(self.parent, text='Hasil dan Plot',bg='white', command=self.display).place(x=30,y=130,height=30)
        self.text = tk.Text(self.parent, height=2, width=30,bg='white')
        self.text.place(x=200,y=130,height=30)
        self.text5 = tk.Text(self.parent, height=2, width=30,bg='white')
        self.text5.place(x=200,y=10,height=30)
        self.text6 = tk.Text(self.parent, height=2, width=30,bg='white')
        self.text6.place(x=200,y=50,height=30)
        self.text7 = tk.Text(self.parent, height=2, width=30,bg='white')
        self.text7.place(x=200,y=90,height=30)
        self.tombol()
        self.fxn()

    def tombol(self):
        self.button = tk.Button(self.parent, text='Tentang GUI&Cara Penggunaan',bg='white', command=self.about).place(x=200,y=180,height=30)
        self.button = tk.Button(self.parent, text='Hapus File',bg='white', command=self.hapus).place(x=250,y=220,height=30)
    def load(self):

        name = askopenfilename(filetypes=[(('*.xls', '*.xlsx'))])

        if name:
            if name.endswith('.xlsx'):
                self.df = pd.read_excel(name)
            self.filename = name
        self.text5.insert('end', ((self.filename)))

    def load1(self):

        name1 = askopenfilename(filetypes=[(('*.xls', '*.xlsx'))])

        if name1:
            if name1.endswith('.xlsx'):
                self.df1 = pd.read_excel(name1, 'Sheet2',converters={'*':str})
            self.filename1 = name1
        self.text6.insert('end', ((self.filename1)))

    def load2(self):

        name2 = askopenfilename(filetypes=[(('*.xls', '*.xlsx'))])

        if name2:
            if name2.endswith('.xlsx'):
                self.df2 = pd.read_excel(name2, 'Sheet3',converters={'*':str})
            self.filename2 = name2
        self.text7.insert('end', ((self.filename1)))
    def display(self):
        if self.df is None:
            self.load()
        # display if loaded
        if self.df is not None:
            data = self.df
            MatriksDataObjek = data.iloc[:]
            MatriksDataVariabel = np.transpose(MatriksDataObjek)
            D1 =euclidean_distances(MatriksDataObjek)
            D2 =euclidean_distances(MatriksDataVariabel)
            n1 = len(D1)
            n2 = len(D2)
            H1 = np.eye(n1) - np.ones((n1, n1))/n1
            H2 = np.eye(n2) - np.ones((n2, n2))/n2
            B1 = -H1.dot(D1**2).dot(H1)/2
            B2 = -H2.dot(D2**2).dot(H2)/2
            evals1, evecs1 = np.linalg.eigh(B1)
            evals2, evecs2 = np.linalg.eigh(B2)
            idx1   = np.argsort(evals1)[::-1]
            idx2   = np.argsort(evals2)[::-1]
            evals1 = evals1[idx1]
            evals2 = evals2[idx2]
            evecs1 = evecs1[:,idx1]
            evecs2 = evecs2[:,idx2]
            w1, = np.where(evals1 > 0)
            w2, = np.where(evals2 > 0)
            L1  = np.diag(np.sqrt(evals1[w1]))
            L2  = np.diag(np.sqrt(evals2[w2]))
            V1  = evecs1[:,w1]
            V2  = evecs2[:,w2]
            Y1  = V1.dot(L1)
            Y2  = V2.dot(L2)
            x1, y1 = Y1[:, 0], Y1[:, 1]
            x2, y2 = Y2[:, 0], Y2[:, 1]
            coor1 = Y1[:, 0], Y1[:, 1]
            coor2 = Y2[:, 0], Y2[:, 1]
            transcoor1= np.transpose(coor1)
            Dcoor= euclidean_distances(transcoor1)
            StressA= np.sum((D1 - Dcoor)**2)
            StressB= np.sum((D1-(D1.sum()/n1))**2)
            STRESS=np.sqrt(StressA/StressB)
        self.text.insert('end', ((STRESS)))
        top = tk.Toplevel(root)
        top.title("Hasil Perhitungan")
        self.scrollbar= tk.Scrollbar(top, orient='vertical')
        self.scrollbar.pack(side='right',fill='y')
        self.text2 = tk.Text(top, background="white",yscrollcommand=self.scrollbar.set)
        self.text2.pack(expand=1, fill="both")
        self.text2.tag_configure('big', font=('Verdana',10,'bold'), foreground= 'blue')
        self.text2.tag_configure('norm', font=('Verdana',10,'bold'), foreground= 'black')
        self.text2.insert('end', 'JARAK OBJEK\n', 'big', D1)
        self.text2.insert('end', '\nMATRIKS B OBJEK\n', 'big', B1)
        self.text2.insert('end', '\nNILAI EIGEN OBJEK\n', 'big', evals1)
        self.text2.insert('end', '\nVEKTOR EIGEN OBJEK\n', 'big', evecs1)
        self.text2.insert('end', '\nKOORDINAT OBJEK\n', 'big')
        self.text2.insert('end', 'x1\n', 'norm', Y1[:, 0])
        self.text2.insert('end', '\ny1\n', 'norm', Y1[:, 1])
        self.text2.insert('end', '\nJARAK KOORDINAT OBJEK\n', 'big', Dcoor)
        self.text2.insert('end', '\nJARAK VARIABEL\n', 'big', D2)
        self.text2.insert('end', '\nMATRIKS B VARIABEL\n', 'big', B2)
        self.text2.insert('end', '\nNILAI EIGEN VARIABEL\n', 'big', evals2)
        self.text2.insert('end', '\nVEKTOR EIGEN VARIABEL\n', 'big', evecs2)
        self.text2.insert('end', '\nKOORDINAT VARIABEL\n', 'big')
        self.text2.insert('end', 'x2\n', 'norm', Y2[:, 0])
        self.text2.insert('end', '\ny2\n', 'norm', Y2[:, 1])
        self.scrollbar.config(command=self.text2.yview)
        if self.df1 is None:
            self.load1()
        if self.df2 is None:
            self.load2()
        if self.df1 is not None:
            if self.df2 is not None:
                NamaObjek=self.df1
                fig1, ax1 = plt.subplots()
                ax1.scatter(x1, y1)
                ax1.set_title('Plot Objek')
                for (Objek1, _x1, _y1) in zip(NamaObjek, x1, y1):
                    ax1.annotate(Objek1, (_x1, _y1), color='green')
                plt.axhline(y=0, xmin=0, xmax=1, linewidth=2, color='k')
                plt.axvline(x=0, ymin=0, ymax=1, linewidth=2, color='k')
                NamaVariabel=self.df2
                fig2, ax2 = plt.subplots()
                ax2.scatter(x2, y2)
                ax2.set_title('Plot Variabel')
                for (Variabel1, _x2, _y2) in zip(NamaVariabel, x2, y2):
                    ax2.annotate(Variabel1, (_x2, _y2), color='blue')
                plt.axhline(y=0, xmin=0, xmax=1, linewidth=2, color='k')
                plt.axvline(x=0, ymin=0, ymax=1, linewidth=2, color='k')
                fig3, ax3 = plt.subplots()
                ax3.scatter(x1, y1)
                ax3.scatter(x2, y2)
                ax3.set_title('Plot Joint')
                for (Objek1, _x1, _y1) in zip(NamaObjek, x1, y1):
                    ax3.annotate(Objek1, (_x1, _y1), color='green')
                for (Variabel1, _x2, _y2) in zip(NamaVariabel, x2, y2):
                    ax3.annotate(Variabel1, (_x2, _y2), color='blue')
                plt.axhline(y=0, xmin=0, xmax=1, linewidth=2, color='k')
                plt.axvline(x=0, ymin=0, ymax=1, linewidth=2, color='k')
            plt.show()
            plt.show()
            plt.show()
    def about(self):
            f = open('d:\cara.txt','r')
            txt_file = f.read()
            top = tk.Toplevel(root)
            top.title("Tentang GUI")
            self.text2 = tk.Text(top, background="white")
            self.text2.pack(expand=1, fill="both")
            self.text2.insert('end', (('[Tentang]=', txt_file)))
    def hapus(self):
            self.text.delete("1.0", "end")
            self.text5.delete("1.0", "end")
            self.text6.delete("1.0", "end")
            self.text7.delete("1.0", "end")
    def fxn(self):
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            
 if __name__ == '__main__':
 
    root = tk.Tk()
    root.resizable(0,0)
    root.title("MULTIDIMENSIONALSCALING METRIK")
    root.geometry("500x300")
    root.iconbitmap(r'd:/unnamed_1_o7R_icon.ico')
    top = MyWindow(root)
    root.mainloop()
