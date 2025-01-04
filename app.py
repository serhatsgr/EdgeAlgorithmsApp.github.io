from flask import Flask, render_template, redirect, url_for
from flask_socketio import SocketIO
import cv2
import base64
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)


cap = cv2.VideoCapture(0)


camera_algorithm1 = 'sobel'
camera_algorithm2 = 'sobel'
photo_algorithm1 = 'sobel'
photo_algorithm2 = 'sobel'
uploaded_image = None 

def apply_algorithm(frame, algorithm):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if algorithm == 'laplacian':
        edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    elif algorithm == 'sobel':
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    elif algorithm == 'canny':
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    
    elif algorithm == 'prewitt':

        prewitt_x = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]])

        prewitt_y = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]])

        edges_x = cv2.filter2D(gray, -1, prewitt_x)
        edges_y = cv2.filter2D(gray, -1, prewitt_y)

   
        edges = cv2.magnitude(edges_x.astype(np.float64), edges_y.astype(np.float64))
    
    #Scharr, Sobel filtresine benzer, ancak daha yüksek bir hassasiyete sahiptir.
    elif algorithm == 'scharr':
        edges_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        edges_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        edges = cv2.magnitude(edges_x.astype(np.float64), edges_y.astype(np.float64)) 

    #Roberts kenar algılama, iki farklı 2D filtre kullanır.
    elif algorithm == 'roberts':
        roberts_x = np.array([[1, 0],
                            [0, -1]])
        roberts_y = np.array([[0, 1],
                            [-1, 0]])

        edges_x = cv2.filter2D(gray, -1, roberts_x)
        edges_y = cv2.filter2D(gray, -1, roberts_y)
        edges = cv2.magnitude(edges_x.astype(np.float64), edges_y.astype(np.float64))   

    #Laplace of Gaussian (LoG)
    #Bu algoritma, görüntüyü önce Gaussian filtre ile yumuşatır, ardından Laplacian uygular.
    elif algorithm == 'log':
        gaussian_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Laplacian(gaussian_blurred, cv2.CV_64F)        


    elif algorithm == 'threshold':
      _, edges = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    #Adaptive Threshold
    #Görüntünün farklı bölgelerine göre uyarlanmış bir eşikleme yöntemi
    elif algorithm == 'adaptive_threshold':
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


    #Frei-Chen  
    # Bu algoritma, çeşitli yönlerde kenarları algılamak için kullanılır
    elif algorithm == 'frei-chen':
        frei_chen = np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]])
        edges = cv2.filter2D(gray, -1, frei_chen)    

    #Hough Transform
    #Hough Transform, doğrusal ve dairesel kenarları algılamak için kullanılır.

    elif algorithm == 'hough_lines':
        edges = cv2.Canny(gray, 100, 200)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        # Çizim için bir kopya oluştur
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        edges = output    

    #Gabor Filter
    elif algorithm == 'gabor':
        gabor_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        edges = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)    
 
    #K-means Clustering for Edge Detection 
    elif algorithm == 'kmeans':
        Z = gray.reshape((-1, 1))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        edges = labels.reshape(gray.shape)
   
   #Directional Derivative Filters
   #Farklı yönlerde kenar algılamak için yönlü türev filtreleri kullanabilirsiniz.
    elif algorithm == 'directional_derivative':
        kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        edges_x = cv2.filter2D(gray, -1, kernel_x)
        edges_y = cv2.filter2D(gray, -1, kernel_y)
        edges = cv2.magnitude(edges_x, edges_y)    

    #Farklı ölçeklerde kenarları algılamak için kullanılabilir.
    elif algorithm == 'multi_scale_gaussian':
        scales = [1, 2, 3]
        edges = np.zeros_like(gray, dtype=np.float64)
        for scale in scales:
            blurred = cv2.GaussianBlur(gray, (0, 0), scale)
            edges += cv2.Laplacian(blurred, cv2.CV_64F)
        edges = cv2.convertScaleAbs(edges)  

    #Image Gradient Magnitude
    #Görüntü gradyanı büyüklüğünü hesaplayarak kenarları belirleyebilirsiniz.
    elif algorithm == 'gradient_magnitude':
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.magnitude(sobel_x, sobel_y)  

    #Morphological Edge Detection
    #Morfolojik işlemlerle kenar algılamak için açma ve kapama işlemleri kullanabilirsiniz.
    elif algorithm == 'morphological':
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel) 
           

        

    elif algorithm == 'gaussian':
        edges = cv2.GaussianBlur(gray, (5, 5), 0)
    elif algorithm == 'median':
        edges = cv2.medianBlur(gray, 5)
    elif algorithm == 'bilateral':
        edges = cv2.bilateralFilter(gray, 9, 75, 75)    
        
    else:
        edges = gray
    return cv2.convertScaleAbs(edges)

@socketio.on('start_camera')
def start_camera(data):
    global camera_algorithm1, camera_algorithm2

    camera_algorithm1 = data.get('algorithm1', 'sobel')
    camera_algorithm2 = data.get('algorithm2', 'sobel')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

     
        edges1 = apply_algorithm(frame, camera_algorithm1)
        edges2 = apply_algorithm(frame, camera_algorithm2)

      
        _, buffer1 = cv2.imencode('.jpg', edges1)
        edge_frame1 = base64.b64encode(buffer1).decode('utf-8')

        _, buffer2 = cv2.imencode('.jpg', edges2)
        edge_frame2 = base64.b64encode(buffer2).decode('utf-8')

        socketio.emit('camera_frame', {'edge_frame1': edge_frame1, 'edge_frame2': edge_frame2})
        socketio.sleep(0.05)  # 20 FPS hızında görüntü almak için



@socketio.on('upload_image')
def upload_image(data):
    global uploaded_image, photo_algorithm1, photo_algorithm2

  
    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    uploaded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    edges1 = apply_algorithm(uploaded_image, photo_algorithm1)
    edges2 = apply_algorithm(uploaded_image, photo_algorithm2)


    _, buffer1 = cv2.imencode('.jpg', edges1)
    edge_frame1 = base64.b64encode(buffer1).decode('utf-8')

    _, buffer2 = cv2.imencode('.jpg', edges2)
    edge_frame2 = base64.b64encode(buffer2).decode('utf-8')

    socketio.emit('uploaded_frame', {'edge_frame1': edge_frame1, 'edge_frame2': edge_frame2})



@socketio.on('change_algorithm')
def change_algorithm(data):
    global photo_algorithm1, photo_algorithm2, uploaded_image


    photo_algorithm1 = data.get('algorithm1', 'sobel')
    photo_algorithm2 = data.get('algorithm2', 'sobel')

  
    if uploaded_image is not None:
        edges1 = apply_algorithm(uploaded_image, photo_algorithm1)
        edges2 = apply_algorithm(uploaded_image, photo_algorithm2)

        _, buffer1 = cv2.imencode('.jpg', edges1)
        edge_frame1 = base64.b64encode(buffer1).decode('utf-8')

        _, buffer2 = cv2.imencode('.jpg', edges2)
        edge_frame2 = base64.b64encode(buffer2).decode('utf-8')

        socketio.emit('uploaded_frame', {'edge_frame1': edge_frame1, 'edge_frame2': edge_frame2})



@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/blog')
def blog():
    return render_template('blogPages/blog.html')


@app.route('/index')
def index():
    return render_template('index.html')

@app.route("/sobel")
def sobel():
    return render_template("blogPages/sobel.html")  

@app.route("/canny")
def canny():
    return render_template("blogPages/canny.html")  

@app.route("/laplacian")
def laplacian():
    return render_template("blogPages/laplacian.html") 

@app.route("/adaptive_threshold")
def adaptive():
    return render_template("blogPages/adaptive_threshold.html")

@app.route("/bilateral")
def bilateral():
    return render_template("blogPages/bilateral.html")

@app.route("/freichen")
def freichen():
    return render_template("blogPages/freichen.html")

@app.route("/gabor")
def gabor():
    return render_template("blogPages/gabor.html")

@app.route("/gaussian")
def gaussian():
    return render_template("blogPages/gaussian.html")

@app.route("/gradient_magnitude")
def gradient():
    return render_template("blogPages/gradient_magnitude.html")

@app.route("/hough_lines")
def hough():
    return render_template("blogPages/hough_lines.html")

@app.route("/log")
def log():
    return render_template("blogPages/log.html")

@app.route("/median")
def median():
    return render_template("blogPages/median.html")

@app.route("/morphological")
def morphological():
    return render_template("blogPages/morphological.html")

@app.route("/prewitt")
def prewitt():
    return render_template("blogPages/prewitt.html")

@app.route("/roberts")
def roberts():
    return render_template("blogPages/roberts.html")

@app.route("/scale_gaussian")
def scalegaussian():
    return render_template("blogPages/scale_gaussian.html")

@app.route("/scharr")
def scharr():
    return render_template("blogPages/scharr.html")

@app.route("/threshold")
def threshold():
    return render_template("blogPages/threshold.html")



@app.route("/start-test")
def start_test():
    return redirect(url_for("index"))  


if __name__ == '__main__':
    socketio.run(app, debug=True)