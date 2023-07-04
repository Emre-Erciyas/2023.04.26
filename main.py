"""
Main.py
"""
import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from ultralytics import YOLO
from tkinter import *
from tkinter import filedialog
from tkcalendar import Calendar
from tkinter import ttk
from datetime import datetime, timedelta
import logging
import threading


# initiliaze firebase
cred = credentials.Certificate(
    "firebaseKey.json")

firebase_admin.initialize_app(cred)

db = firestore.client()

# load the pre-trained model
model = YOLO("yolov8x.pt")

# FUNCTIONS ----------------------------------------------------------------------


# fetch places from the database and return it as a list
def fetchPlaces():
    temp = []
    docs = db.collection(u'places').stream()
    for doc in docs:
        temp.append(f'{doc.id}')
    return temp


# take video start date and time as input. temporary solution
def getVideoStartTime(place, frame):

    # path file to video
    file_path = filedialog.askopenfilename()
    if file_path == "":
        return
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(frame)
    newWindow.title("Set video start time")
    newWindow.geometry("1200x800")

    # start date input
    cal = Calendar(newWindow,
                   font="Arial 14", selectmode='day',
                   cursor="hand1", year=2023, month=5, day=14)

    cal.pack(fill="both", expand=True, padx=50, pady=20)
    s = ttk.Style(newWindow)
    s.theme_use('clam')

    # start time input
    startLastValue = ""

    def traceStart(*args):
        nonlocal startLastValue
        if startLastValue == "59" and startMinString() == "0":
            startHourString.set(int(startHourString.get()) +
                                1 if startHourString.get() != "23" else 0)
        startLastValue = startMinString.get()

    HMframe = Frame(newWindow)
    start = Label(HMframe, text='Video Start Time:', font=("Arial", 15))
    start.grid(row=0, column=0, padx=(10), sticky='W')
    startHourString = StringVar(HMframe, '10')
    startHour = Spinbox(HMframe, from_=0, to=23, wrap=True,
                        textvariable=startHourString, width=2, state="readonly")
    startMinString = StringVar(HMframe, '30')
    startMinString.trace("w", traceStart)
    startMin = Spinbox(HMframe, from_=0, to=59, wrap=True,
                       textvariable=startMinString, width=2, state="readonly")

    startHour.grid(row=0, column=1)
    startMin.grid(row=0, column=2)

    HMframe.pack(pady=(50, 0))

    # end window
    def endWindow():
        year = int(cal.selection_get().year)
        month = int(cal.selection_get().month)
        day = int(cal.selection_get().day)
        # year, month, day = map(int, cal.selection_get().split('-'))
        date = datetime(year, month, day, int(
            startHourString.get()), int(startMinString.get()), 0)
        newWindow.destroy()
        processVideo(place, date, file_path)

    Button(newWindow, text="Done", command=endWindow).pack(pady=50)


def write_to_database(place, currentDate, total):
    dateDB = currentDate.strftime('%Y-%m-%d')
    timeDB = currentDate.strftime('%H:%M')
    result = float(format(total / 60, ".2f"))
    data = {
        timeDB: result
    }
    db.collection(u'places').document(place).collection(
        'records').document(dateDB).set(data, merge=True)


# takes start time and file path and predicts how many people are in the frame.
# Then writes this information to the database every minute.
def processVideo(place, startDate, file_path):
    # Open the video file
    video = cv2.VideoCapture(file_path)

    # Check if the video file was opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return

    videoDuration = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = int(video.get(cv2.CAP_PROP_FPS))

    i = 0
    total = 0
    x = 0
    currentDate = startDate
    for i in range(videoDuration):
        # Read the next frame
        ret, frame = video.read()

        # only read one frame a second so that it is not extremely slow
        if (i % FPS != 0):
            continue

        # If the frame was not read successfully, break out of the loop
        if not ret:
            break

        # model predicts
        results = model.predict(
            frame, stream=True, conf=0.15, classes=0, verbose=False)

        if x >= 60:
            x -= 60
            write_thread = threading.Thread(
                target=write_to_database, args=(place, currentDate, total))
            write_thread.start()
        for result in results:
            # number of boxes are the number of people
            total += int(result.boxes.shape[0])
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2),
                                                          int(y2)), (255, 0, 255), 5)

        # Display the frame with bounding boxes
        cv2.imshow('Frame ' + str(i), frame)

        # press q to exit
        if cv2.waitKey(75) & 0xFF == ord('q'):
            break

        x += 1
        currentDate += timedelta(seconds=1)

    # Release the video capture object and close all windows
    video.release()
    cv2.destroyAllWindows()


def addEvent(place, frame):
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(frame)
    newWindow.title("Event Details")
    newWindow.geometry("1200x800")

    # Event name input
    eventName = Entry(newWindow, bg='white', fg='black')
    eventName.insert(0, "Event name")
    eventName.pack(pady=30)

    # start date, end date labels
    labelFrame = Frame(newWindow)
    startLabel = Label(labelFrame, text='Start Date:', font=("Arial", 25))
    endLabel = Label(labelFrame, text='End Date:', font=("Arial", 25))
    startLabel.pack(side=LEFT, padx=15)
    endLabel.pack(padx=15)
    labelFrame.pack(fill='both', padx=40)

    # Calendars
    calendarFrame = Frame(newWindow)
    calStart = Calendar(calendarFrame,
                        font="Arial 14", selectmode='day',
                        cursor="hand1", year=2023, month=5, day=14)

    calStart.pack(fill="both", expand=True, side=LEFT, padx=15)
    calEnd = Calendar(calendarFrame,
                      font="Arial 14", selectmode='day',
                      cursor="hand1", year=2023, month=5, day=14)

    calEnd.pack(fill="both", expand=True, side=RIGHT, padx=15)
    calendarFrame.pack(fill="both", expand=True, pady=20, padx=40)
    s = ttk.Style(newWindow)
    s.theme_use('clam')

    # Time inputs
    startLastValue = ""
    endLastValue = ""

    def traceStart(*args):
        nonlocal startLastValue
        if startLastValue == "59" and startMinString() == "0":
            startHourString.set(int(startHourString.get()) +
                                1 if startHourString.get() != "23" else 0)
        startLastValue = startMinString.get()

    def traceEnd(*args):
        nonlocal endLastValue
        if endLastValue == "59" and endMinString() == "0":
            endHourString.set(int(endHourString.get()) +
                              1 if endHourString.get() != "23" else 0)
        endLastValue = endMinString.get()
    hoursFrame = Frame(newWindow)
    HMframeStart = Frame(hoursFrame)
    start = Label(HMframeStart, text='Start: ', font=("Arial", 15))
    start.grid(row=0, column=0, padx=(
        20, 20), sticky='W')
    startHourString = StringVar(HMframeStart, '10')
    startHour = Spinbox(HMframeStart, from_=0, to=23, wrap=True,
                        textvariable=startHourString, width=2, state="readonly")
    startMinString = StringVar(HMframeStart, '30')
    startMinString.trace("w", traceStart)
    startMin = Spinbox(HMframeStart, from_=0, to=59, wrap=True,
                       textvariable=startMinString, width=2, state="readonly")

    startHour.grid(row=0, column=1)
    startMin.grid(row=0, column=2)
    HMframeEnd = Frame(hoursFrame)
    end = Label(HMframeEnd, text='End: ', font=("Arial", 15))
    end.grid(row=0, column=3, padx=(
        20, 20), sticky='W')
    endHourString = StringVar(HMframeEnd, '10')
    endHour = Spinbox(HMframeEnd, from_=0, to=23, wrap=True,
                      textvariable=endHourString, width=2, state="readonly")
    endMinString = StringVar(HMframeEnd, '30')
    endMinString.trace("w", traceEnd)
    endMin = Spinbox(HMframeEnd, from_=0, to=59, wrap=True,
                     textvariable=endMinString, width=2, state="readonly")
    endHour.grid(row=0, column=4)
    endMin.grid(row=0, column=5)
    HMframeStart.pack(side=LEFT)
    HMframeEnd.pack()
    hoursFrame.pack(fill='both', padx=40)

    def endWindow():
        data = {
            u'startDate': str(calStart.selection_get()),
            u'endDate': str(calEnd.selection_get()),
            u'startTime': f"{str(startHourString.get()).zfill(2)}:{str(startMinString.get()).zfill(2)}",
            u'endTime': f"{str(endHourString.get()).zfill(2)}:{str(endMinString.get()).zfill(2)}"
        }
        try:
            db.collection(u'places').document(place).collection(
                u'events').document(eventName.get()).set(data)
        except:
            pass
        newWindow.destroy()
    Button(newWindow, text="Create", command=endWindow).pack(pady=(80, 30))


def createPlaces(places, frame):
    for index, place in enumerate(places):
        label = Label(frame, text=place)
        label.grid(row=index, column=0, padx=(
            20, 20), pady=(50, 0), sticky='W')

        button = Button(
            frame, text=f"Add video to {place}", command=lambda place=place: getVideoStartTime(place, frame), cursor="hand2")
        button.grid(row=index, column=1, padx=(
            20, 20), pady=(50, 0), sticky='ew')

        button = Button(
            frame, text=f"Add event to {place}", command=lambda place=place: addEvent(place, frame))
        button.grid(row=index, column=2, padx=(
            20, 20), pady=(50, 0), sticky='ew')

        button = Button(
            frame, text=f"Show events", command=lambda place=place:  showEvents(place, frame), cursor="hand2")
        button.grid(row=index, column=3, padx=(
            20, 20), pady=(50, 0), sticky='ew')

        button = Button(
            frame, text=f"-", command=lambda place=place:  removePlace(place, frame), cursor="hand2")
        button.grid(row=index, column=4, padx=(
            20, 0), pady=(50, 0), sticky='ew')


def removePlace(place, frame):
    def deleteCollection(coll_ref, batch_size):
        docs = coll_ref.list_documents(page_size=batch_size)
        deleted = 0

        for doc in docs:
            doc.delete()
            deleted = deleted + 1

        if deleted >= batch_size:
            return deleteCollection(coll_ref, batch_size)
    try:
        deleteCollection(db.collection(u'places').document(
            place).collection(u'records'), 50)
        deleteCollection(db.collection(u'places').document(
            place).collection(u'events'), 50)
        db.collection(u'places').document(place).delete()
        for child in frame.winfo_children():
            child.destroy()
        createPlaces(fetchPlaces(), frame)
    except:
        pass


def addPlace(place, frame):
    try:
        db.collection(u'places').document(place).set({})
        for child in frame.winfo_children():
            child.destroy()
        createPlaces(fetchPlaces(), frame)
    except:
        pass


def removeEvent(place, eventName, newWindow, frame):
    try:
        db.collection(u'places').document(place).collection(
            u'events').document(eventName).delete()
    except:
        pass
    newWindow.destroy()
    showEvents(place, frame)


def showEvents(place, frame):
    try:
        eventDocs = db.collection(u'places').document(
            place).collection('events').stream()
        newWindow = Toplevel(frame)
        newWindow.title("Events")
        newWindow.geometry("1200x800")

        i = 0
        for eventDoc in eventDocs:
            sd = eventDoc.get('startDate')
            st = eventDoc.get('startTime')
            ed = eventDoc.get('endDate')
            et = eventDoc.get('endTime')
            startYear, startMonth, startDay = map(int, sd.split('-'))
            startHour, startMinute = map(int, st.split(':'))
            startDate = datetime(startYear, startMonth,
                                 startDay, startHour, startMinute)
            endYear, endMonth, endDay = map(int, sd.split('-'))
            endHour, endMinute = map(int, et.split(':'))
            endDate = datetime(endYear, endMonth, endDay, endHour, endMinute)
            recordDocs = db.collection(u'places').document(
                place).collection(u'records').stream()

            total = 0
            for recordDoc in recordDocs:
                if not (recordDoc.id >= sd and recordDoc.id <= ed):
                    continue

                recordYear, recordMonth, recordDay = map(
                    int, recordDoc.id.split('-'))

                recordDict = recordDoc.to_dict()
                for key in recordDict:
                    recordHour, recordMinute = map(int, key.split(':'))
                    recordDate = datetime(
                        recordYear, recordMonth, recordDay, recordHour, recordMinute)
                    if recordDate >= startDate and recordDate <= endDate:
                        total += recordDict[key]

            if not total == 0:
                totalMins = (endDate - startDate).total_seconds() // 60
                if not totalMins <= 0:
                    total = total / totalMins

            eventName = Label(newWindow, text=eventDoc.id)
            eventName.grid(padx=(
                20, 20), pady=(50, 0), column=0, row=i)
            sdLabel = Label(newWindow, text=(f'Start Date:\n\n{sd}'))
            sdLabel.grid(padx=(
                20, 20), pady=(50, 0), column=1, row=i)
            stLabel = Label(newWindow, text=f'Start Time:\n\n{st}')
            stLabel.grid(padx=(
                20, 20), pady=(50, 0), column=2, row=i)
            edLabel = Label(newWindow, text=f'End Date:\n\n{ed}')
            edLabel.grid(padx=(
                20, 20), pady=(50, 0), column=3, row=i)
            etLabel = Label(newWindow, text=f'End Time:\n\n{et}')
            etLabel.grid(padx=(
                20, 20), pady=(50, 0), column=4, row=i)
            totalLabel = Label(
                newWindow, text=f'People per minute:\n\n{"{:.2f}".format(total)}')
            totalLabel.grid(padx=(
                20, 20), pady=(50, 0), column=5, row=i)
            removeEventButton = Button(newWindow, text=f"-", cursor="hand2",
                                       command=lambda eventName=eventDoc.id: removeEvent(place, eventName, newWindow, frame))
            removeEventButton.grid(padx=(
                20, 20), pady=(50, 0), row=i, column=6)

            i += 1

    except Exception as e:
        logging.exception("An exception occurred: %s", str(e))


def showLast(frame):
    newWindow = Toplevel(frame)
    newWindow.title("Search")
    newWindow.geometry("1200x800")

    # start date, end date labels
    labelFrame = Frame(newWindow)
    startLabel = Label(labelFrame, text='Search Start:', font=("Arial", 25))
    endLabel = Label(labelFrame, text='Search End:', font=("Arial", 25))
    startLabel.pack(side=LEFT, padx=15, pady=20)
    endLabel.pack(padx=15, pady=20)
    labelFrame.pack(fill='both', padx=40)

    # Calendars
    calendarFrame = Frame(newWindow)
    calStart = Calendar(calendarFrame,
                        font="Arial 14", selectmode='day',
                        cursor="hand1", year=2023, month=5, day=14)

    calStart.pack(fill="both", expand=True, side=LEFT, padx=15)
    calEnd = Calendar(calendarFrame,
                      font="Arial 14", selectmode='day',
                      cursor="hand1", year=2023, month=5, day=14)

    calEnd.pack(fill="both", expand=True, side=RIGHT, padx=15)
    calendarFrame.pack(fill="both", expand=True, pady=20, padx=40)
    s = ttk.Style(newWindow)
    s.theme_use('clam')

    # Time inputs
    startLastValue = ""
    endLastValue = ""

    def traceStart(*args):
        nonlocal startLastValue
        if startLastValue == "59" and startMinString() == "0":
            startHourString.set(int(startHourString.get()) +
                                1 if startHourString.get() != "23" else 0)
        startLastValue = startMinString.get()

    def traceEnd(*args):
        nonlocal endLastValue
        if endLastValue == "59" and endMinString() == "0":
            endHourString.set(int(endHourString.get()) +
                              1 if endHourString.get() != "23" else 0)
        endLastValue = endMinString.get()
    hoursFrame = Frame(newWindow)
    HMframeStart = Frame(hoursFrame)
    start = Label(HMframeStart, text='Start: ', font=("Arial", 15))
    start.grid(row=0, column=0, padx=(
        20, 20), sticky='W')
    startHourString = StringVar(HMframeStart, '10')
    startHour = Spinbox(HMframeStart, from_=0, to=23, wrap=True,
                        textvariable=startHourString, width=2, state="readonly")
    startMinString = StringVar(HMframeStart, '30')
    startMinString.trace("w", traceStart)
    startMin = Spinbox(HMframeStart, from_=0, to=59, wrap=True,
                       textvariable=startMinString, width=2, state="readonly")

    startHour.grid(row=0, column=1)
    startMin.grid(row=0, column=2)
    HMframeEnd = Frame(hoursFrame)
    end = Label(HMframeEnd, text='End: ', font=("Arial", 15))
    end.grid(row=0, column=3, padx=(
        20, 20), sticky='W')
    endHourString = StringVar(HMframeEnd, '10')
    endHour = Spinbox(HMframeEnd, from_=0, to=23, wrap=True,
                      textvariable=endHourString, width=2, state="readonly")
    endMinString = StringVar(HMframeEnd, '30')
    endMinString.trace("w", traceEnd)
    endMin = Spinbox(HMframeEnd, from_=0, to=59, wrap=True,
                     textvariable=endMinString, width=2, state="readonly")
    endHour.grid(row=0, column=4)
    endMin.grid(row=0, column=5)
    HMframeStart.pack(side=LEFT)
    HMframeEnd.pack()
    hoursFrame.pack(fill='both', padx=40)

    def endWindow():

        start_hour = int(startHourString.get())
        start_minute = int(startMinString.get())
        end_hour = int(endHourString.get())
        end_minute = int(endMinString.get())
        startStr = str(calStart.selection_get())
        endStr = str(calEnd.selection_get())

        records = db.collection_group(u'records').stream()
        events = db.collection_group(u'events').stream()

        searchResultWindow = Toplevel(newWindow)
        searchResultWindow.title("Records")
        searchResultWindow.geometry("1200x800")
        recordText = Text(searchResultWindow)
        eventText = Text(searchResultWindow)
        recordText.insert(END, u'Records:\n\n')
        eventText.insert(END, U'Events:\n\n')
        recordText.pack()
        eventText.pack()
        for record in records:
            recordDate = record.id
            place = record.reference.parent.parent.id
            recordDict = record.to_dict()
            # Insert the dictionary content into the Text widget
            if not (recordDate >= startStr and recordDate <= endStr):
                continue
            for key, value in recordDict.items():
                hour, minute = map(int, key.split(':'))
                if recordDate == startStr and (start_hour * 60 + start_minute) > (hour * 60 + minute):
                    continue
                elif recordDate == endStr and (hour * 60 + minute) > (end_hour * 60 + end_minute):
                    continue

                recordText.insert(
                    END, f"{place} => Date: {recordDate} Time: {key}: {value} People\n")

        for event in events:
            eventDict = event.to_dict()
            eventName = event.id
            place = event.reference.parent.parent.id

            event_start_date = eventDict['startDate']
            event_end_date = eventDict['endDate']
            event_start_time = eventDict['startTime']
            event_end_time = eventDict['endTime']
            if event_start_date > endStr or event_end_date < startStr:
                continue

            hourStart, minuteStart = map(int, event_start_time.split(':'))
            hourEnd, minuteEnd = map(int, event_end_time.split(':'))
            if event_start_date == endStr and (end_hour * 60 + end_minute) < (hourStart * 60 + minuteStart):
                continue
            elif event_end_date == startStr and (hourEnd * 60 + minuteEnd) > (start_hour * 60 + start_minute):
                continue
            eventText.insert(END, f"\n{place}, {eventName} =>\n\n")
            eventText.insert(
                END, f"\tStart: {event_start_date} --> {event_start_time}\n\tEnd:   {event_end_date} --> {event_end_time}\n")
        recordText.configure(state="disabled")
        eventText.configure(state='disabled')
        Button(searchResultWindow, text="Done",
               command=searchResultWindow.destroy).pack(pady=(80, 30))

    buttonFrame = Frame(newWindow)
    buttonFrame.pack(pady=(80, 30))
    Button(buttonFrame, text="Search", command=endWindow).pack(side=LEFT)
    Button(buttonFrame, text="Quit",
           command=lambda: newWindow.destroy()).pack(side=LEFT)


def initializeUI():

    # Tkinter root
    root = Tk()
    root.title('People Counter')
    root.geometry("1920x1080")

    # Title
    label = Label(root, text='Dashboard', font=("Arial", 25))
    label.grid(row=0, column=0, padx=(
        20, 20), pady=(10, 0), sticky='W')
    # Row to add a new place
    newPlaceInput = Entry(root, bg='white', fg='black')
    newPlaceInput.insert(0, "New place")
    newPlaceInput.grid(row=2, column=0, pady=50, padx=(20, 0), sticky='W')
    searchButton = Button(
        root, text="Search", cursor="hand2",
        command=lambda: showLast(root))
    searchButton.grid(row=0, column=0, sticky='E', pady=(10, 0))

    addButton = Button(
        root, text=f"+", cursor="hand2",
        command=lambda: addPlace(newPlaceInput.get(), frame))
    addButton.grid(row=2, column=0, sticky='E')

    # Frame that contains places
    frame = Frame(root)
    frame.grid(row=1)

    return frame, root


def initiliazeApp():

    frame, root = initializeUI()

    createPlaces(fetchPlaces(), frame)

    root.mainloop()

# END OF FUNCTIONS ----------------------------------------------------------------------


initiliazeApp()
