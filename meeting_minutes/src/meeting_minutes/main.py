#importing libraries in our program
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start      
from openai import OpenAI
from pydub import AudioSegment          
from pydub.utils import make_chunks               
from pathlib import Path

from crews.meeting_minutes_crew.meeting_minutes_crew import MeetingMinutesCrew
from crews.gmailcrew.gmailcrew import GmailCrew

import agentops
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

#a container to store transcript and meeting minutes 
class MeetingMinutesState(BaseModel):       #a class inhereting properties of a basemodel of pydantic
                                            #Pydantic's Basemodel ensures structure and validation to "state" of MeetingMinutes flow.
                                            ###Validation in Pydantic means automatically checking that the data you assign to a model follows the expected data types and constraints. If the data does not match the expected format, Pydantic will raise an error.
   
    transcript: str = ""          #to stores transcript of the meeting as a string
    meeting_minutes: str = ""     #to stores formatted meeting minutes of the meeting


class MeetingMinutesFlow(Flow[MeetingMinutesState]):

    @start()                        #used in arewai to mark the start of the flow
    def transcribe_meeting(self):
        print("Generating Transcription")

        SCRIPT_DIR = Path(__file__).parent                #file path for the .wav file
        audio_path = str(SCRIPT_DIR / "EarningsCall.wav")
        
        # Load the audio file
        audio = AudioSegment.from_file(audio_path, format="wav")
        
        # Define chunk length in milliseconds (e.g., 1 minute = 60,000 ms)
        chunk_length_ms = 60000                                 #chunking audio is necessary because-
                                                                #1.file size limit restriction 2.better accuracy 3.avoids timeout errors
        chunks = make_chunks(audio, chunk_length_ms)

        # Iterate over each chunk
        full_transcription = ""
        for i, chunk in enumerate(chunks):
            print(f"Transcribing chunk {i+1}/{len(chunks)}")
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            
            with open(chunk_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",                    ##an openai model to convert audio into text
                    file=audio_file
                )
                full_transcription += transcription.text + " "   ##combined transcript

        self.state.transcript = full_transcription        ##self.state is the instance/object of the class meetingminutes- state is automatically created. It is like what is the "state" of the work.It keeps track/status of the workflow
        print(f"Transcription: {self.state.transcript}")  ##we could have self.trancript as object to store data but we cannot because we are using flow in this program therefore we a using self.state.transcript
                                                           ##self.trancript or self.state.transcript is value of storing values 
    @listen(transcribe_meeting)         #will execute only after @start function has executed
    def generate_meeting_minutes(self):
        print("Generating Meeting Minutes")

        crew = MeetingMinutesCrew()

        inputs = {
            "transcript": self.state.transcript
        }
        meeting_minutes = crew.crew().kickoff(inputs)
        # Convert CrewOutput to string by accessing its value
        self.state.meeting_minutes = str(meeting_minutes)

    @listen(generate_meeting_minutes)
    def create_draft_meeting_minutes(self):
        print("Creating Draft Meeting Minutes")

        crew = GmailCrew()

        inputs = {
            "body": self.state.meeting_minutes  # Now self.state.meeting_minutes is a string
        }

        draft_crew = crew.crew().kickoff(inputs)
        print(f"Draft Crew: {draft_crew}")


def kickoff():
    session = agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))

    meeting_minutes_flow = MeetingMinutesFlow()
    meeting_minutes_flow.plot()
    meeting_minutes_flow.kickoff()

    session.end_session()

if __name__ == "__main__":
    kickoff()