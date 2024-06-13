
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{'databaseURL':"https://faceface-f5d7f-default-rtdb.firebaseio.com/"})
ref=db.reference('Students')
data={
    "1122":{
        "name":"ronaldo",
        "major":"robotics",
        "starting_years":2005,
        "total_attendance":7,
        "last_attendance":'2024-4-8 12:21:23'
    },
"3344":{
        "name":"elon mask",
        "major":"space",
        "starting_years":2012,

        "total_attendance": 90,

        "last_attendance":'2024-01-4 00:11:23'
    },
    "5566":{
        "name":"nayemar",
        "major":"foteball player",
        "starting_years":2000,
        "total_attendance": 7,
        "last_attendance":'2024-7-9 15:12:23'
    },
    "7788":{
        "name":"mohamad",
        "major":"programmer",
        "starting_years":2005,
        "total_attendance": 7,
        "last_attendance":'2023-7-4 14:19:23'
    }
}
for key,value in data.items():
    ref.child(key).set(value)