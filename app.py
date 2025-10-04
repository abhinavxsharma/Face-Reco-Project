# Import necessary libraries                                                                                #ACTJA1 - 2 
import cv2  # OpenCV library for computer vision tasks
import os  # Operating system related functions
from flask import Flask, request, render_template, redirect  # Flask for web application
from datetime import date, datetime  # Date and time manipulation
import numpy as np  # Numerical operations with arrays
from sklearn.neighbors import KNeighborsClassifier  # Machine learning model for face recognition
import pandas as pd  # Data manipulation with dataframes
from fpdf import FPDF  # Create PDFs
import joblib  # Load and save machine learning models
import shutil  # File operations
import smtplib  # Email sending functionality
from email.mime.multipart import MIMEMultipart  # Email message composition
from email.mime.text import MIMEText  # Email text content
from email.mime.application import MIMEApplication  # Email attachments

# Create a Flask web application instance
app = Flask(__name__)

# Initialize a face detector using the Haar Cascade classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get the current date in two different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Create necessary directories if they don't exist
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('face data'):
    os.makedirs('face data')
if not os.path.isdir('face data/faces'):
    os.makedirs('face data/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Class,Section,Time')

# Define a function to get the total number of registered users
def totalreg():
    return len(os.listdir('face data/faces'))

# Define a function to extract faces from an image
def extract_faces(img):
    try:
        if img.shape != (0, 0, 0):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_points = face_detector.detectMultiScale(gray, 1.3, 5)
            return face_points
        else:
            return []
    except:
        return []

# Define a function to identify a face using an ML model
def identify_face(facearray):
    model = joblib.load('face data/face_recognition_model.pkl')
    return model.predict(facearray)

# Define a function to train the ML model using available face data
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('face data/faces')
    for user in userlist:
        for imgname in os.listdir(f'face data/faces/{user}'):
            img = cv2.imread(f'face data/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'face data/face_recognition_model.pkl')

# Define a function to extract attendance information from a CSV file
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    clas = df['Class']
    section = df['Section']
    l = len(df)
    return names, rolls, times, clas, section, l

# Define a function to add attendance for a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1] 
    userclas = name.split('_')[2]
    newusersec = name.split('_')[3]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{userclas},{newusersec},{current_time}')




# Define the login route
@app.route('/login')
def login():
    return render_template('LOGIN.html')

# Define the home route
@app.route('/')
def home():
    # Extract attendance data
    names, rolls, times, clas, section, l = extract_attendance()
    
    # Render the 'home.html' template with attendance and registration data
    return render_template('home.html', names=names, rolls=rolls, times=times, clas=clas,  section=section, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Code for sending an email with the attendance report
@app.route('/send_email', methods=['POST'])
def send_email():
    try:
        # Set up the SMTP server
        smtp_server = "smtp.mailgun.org"
        smtp_port = 587
        smtp_username = "postmaster@sandbox6490b9f056c643e099f4896a8d3ec67b.mailgun.org"
        smtp_password = "7747d97511a92d1d57b69cd6e5df3bff-451410ff-b35e8a40"
   
        sender_email = "vats.abhinav247@gmail.com"
        receiver_email = "sunilvats1981@gmail.com"
        subject = "Attendance Report"
        message = "Hello, here is the attendance report."

        # Connect to the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)

        current_date = date.today().strftime("%m_%d_%y")
        pdf_filename = f"ATTENDANCE PDF'S/Attendance-{current_date}.pdf"    

        # Read the PDF file as binary data
        with open(pdf_filename, 'rb') as pdf_file:
            pdf_data = pdf_file.read()

        # Compose the email
        email = MIMEMultipart()
        email['From'] = sender_email
        email['To'] = receiver_email
        email['Subject'] = subject
        email.attach(MIMEText(message))  # Add text message
        email.attach(MIMEApplication(pdf_data, _subtype="pdf"))  # Add PDF attachment

        # Send the email
        server.sendmail(sender_email, receiver_email, email.as_string())
        server.quit()

        return f"Email sent successfully to {receiver_email}"
    except Exception as e:
        # Log the error message
        print("Error sending email:", e)
        return "Failed to send email.", 500  # Return error message and 500 status code

# This function runs when you click on the Take Attendance button
# This function runs when you click on the Take Attendance button
@app.route('/start', methods=['GET'])
def start():
    # Check if the trained face recognition model exists in the 'face data' folder
    if 'face_recognition_model.pkl' not in os.listdir('face data'):
        # Render 'home.html' template with an error message if the model is not found
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the face data folder. Please add a new face to continue.')

    # Check if there are any registered faces
    if totalreg() == 0:
        # Render 'home.html' template with an error message indicating no registered faces
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='No faces have been registered for recognition. Please add a new face to continue.')

    # Open the computer's camera for capturing images
    cap = cv2.VideoCapture(0)
    ret = True

    while ret:
        ret, frame = cap.read()
        detected_faces = extract_faces(frame)
        if len(detected_faces) > 0:
            (x, y, w, h) = detected_faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        else:
            # If no faces are detected, display an "Unknown Face" message
            cv2.putText(frame, 'Unknown Face', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    # Release the camera and close all OpenCV windows
    cv2.destroyAllWindows()

    names, rolls, times, clas, section, l = extract_attendance()

    return render_template('home.html', names=names, rolls=rolls, times=times, clas=clas, section=section, l=l, totalreg=totalreg(), datetoday2=datetoday2)



# This function runs when you add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    newuserclas = request.form['newuserclas']
    newusersec = request.form['newusersec']
    userimagefolder = 'face data/faces/' + newusername + '_' + str(newuserid) + '_' +  str(newuserclas) + '_'  + str(newusersec)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Images Captured --  {i}/50', (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 23, 0), 2, cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + newuserclas + '_' + newusersec + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, clas, section, l = extract_attendance()    
    return render_template('home.html', names=names, rolls=rolls, times=times, clas=clas,  section=section,l=l, totalreg=totalreg(), datetoday2=datetoday2) 

##########################################################DELETION CODE'S##########################################################################
# Function to extract user information (name and user_id) from a filename
def extract_info_from_filename(filename):
    parts = filename.split('_')
    name = parts[0]
    user_id = parts[1]
    user_class = parts[2]
    user_section = parts[3]
    return name, user_id , user_class  , user_section




# Route to list users and allow user deletion
@app.route('/list_users')
def list_users():
    faces_directory = "face data/faces"
    table_data = generate_table_data(faces_directory)
    return render_template('list_users.html', table_data=table_data)

# Route to handle user deletion
# Define a route that listens for POST requests to '/delete_user'
@app.route('/delete_user', methods=['POST'])
def delete_user_handler():
    # Specify the directory where user faces are stored
    faces_directory = "face data/faces"
    
    # Get the 'name', 'user_id', 'user_class', and 'user_section' values from the POST request form
    name = request.form.get('name')
    user_id = request.form.get('user_id')
    user_class = request.form.get('user_class')
    user_section = request.form.get('user_section')
    
    # Get the entered password from the form
    entered_password = request.form.get('password')
    
    # Perform password validation here (replace 'your_password' with the actual password)
    if entered_password == 'password':
        # If the password is correct, call the 'delete_user' function
        delete_user(name, user_id, user_class, user_section, faces_directory)
        
        # Redirect the user to the '/list_users' route after deletion
        return redirect('/list_users')
    else:
        # If the password is incorrect, display an error message
        return render_template('list_users.html', table_data=generate_table_data(faces_directory), password_error=True)

# Route to generate a PDF report of the attendance
@app.route('/get_pdf', methods=['GET'])
def get_pdf():
    # Get attendance data
    names, rolls, times, clas , section, l = extract_attendance()

    # Create a PDF object
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()

    # Set font
    pdf.set_font("Arial", size=12)

    # Add attendance data to the PDF
    pdf.cell(0, 10, txt="Attendance Report", ln=True, align='C')
    pdf.cell(0, 10, txt=f"Date: {datetoday2}", ln=True, align='C')
    pdf.cell(0, 10, txt="", ln=True)  # Empty line

    # Add table headers
    pdf.cell(20, 10, "S No", border=1)
    pdf.cell(50, 10, "Name", border=1)
    pdf.cell(20, 10, "ID", border=1)
    pdf.cell(40, 10, "Time", border=1)
    pdf.cell(40,10, "Class" , border=1)
    pdf.cell(20,10, "Section" , border=1)
    pdf.ln()

    # Add attendance data to the table
    for i in range(l):
        pdf.cell(20, 10, str(i + 1), border=1)
        pdf.cell(50, 10, names[i], border=1)
        pdf.cell(20, 10, str(rolls[i]), border=1)
        pdf.cell(40, 10, times[i], border=1)
        pdf.cell(40,10 , str(clas[i]) , border=1)
        pdf.cell(20,10 , str(section[i]) , border=1)
        pdf.ln()

    # Save the PDF to a file
    pdf_file_path = f"ATTENDANCE PDF'S/Attendance-{datetoday}.pdf"
    pdf.output(pdf_file_path)

    # Return a message with a link to the generated PDF
    return f"Attendance PDF generated. <a href='{pdf_file_path}' target='_blank'>Open PDF</a>"

# Function to generate data for listing users in a table
def generate_table_data(directory):
    table_data = []
    if os.path.exists(directory) and os.path.isdir(directory):
        files = os.listdir(directory)
        for idx, file in enumerate(files, start=1):
            name, user_id, user_class, user_section = extract_info_from_filename(file)
            table_data.append((idx, name, user_id , user_class , user_section))
    return table_data

# Function to delete a user's folder and images
def delete_user(name, user_id, user_class, user_section, directory):
    user_folder = f"{name}_{user_id}_{user_class}_{user_section}"
    user_directory = os.path.join(directory, user_folder)
    if os.path.exists(user_directory):
        try:
            shutil.rmtree(user_directory)
            print(f"User {name} with ID {user_id} in  Class {user_class} of CSection {user_section} has been deleted.")
        except Exception as e:
            print(f"Error while deleting user {name} with ID {user_id} class{user_class} of CSection{user_section}: {e}")
    else:
        print(f"User {name} with ID {user_id} in class{user_class} of CSection{user_section} not found.")

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
