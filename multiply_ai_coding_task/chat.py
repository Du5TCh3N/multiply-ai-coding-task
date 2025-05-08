import os
from dataclasses import dataclass, field
import json
import re
from typing import Optional, Dict, Any
import datetime as dt
from dotenv import load_dotenv
from enum import Enum
from .factfind import User, Goal, GoalType, NewHomeGoalInformation, NewCarInformation, OtherGoalInformation
from google import genai
from google.api_core.exceptions import GoogleAPIError

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def llm(prompt: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )
        return response.text
    except GoogleAPIError as e:
        print(f"Error calling Gemini API: {e}")
        return None

STOP_COMMANDS = {"exit", "quit", "stop", "done"}

class Sender(Enum):
    USER = "user"
    AI = "ai"


@dataclass
class Message:
    text: str
    sender: Sender

class ConversationStage(Enum):
    GET_USER_INFO = "get_user_info" # Collect basic personal details (name, DOB, email)
    GET_GOAL_TYPE = "get_goal_type" # Ask if its home/car/other
    GET_GOAL_DETAILS = "get_goal_details" # Collect goal type specific information
    CONFIRM_GOAL = "confirm_goal" # ask yes or no
    ADD_ANOTHER_GOAL = "add_another_goal" # ask yes or no
    COMPLETED = "completed" # End the conversation with stop commands

@dataclass
class ExtractedInformation:
    user: Optional[User] = None
    pending_goal: Optional[Goal] = None
    conversation_stage: ConversationStage = ConversationStage.GET_USER_INFO
    
    def __str__(self) -> str:
        if not self.user:
            return "No user information collected yet."
            
        user_info = f"User: {self.user.first_name} {self.user.last_name}, Email: {self.user.email}, DOB: {self.user.date_of_birth}"
        goals_info = []
        
        for i, goal in enumerate(self.user.goals):
            goal_str = f"Goal {i+1}: {goal.goal_name} ({goal.goal_type.value})"
            
            if isinstance(goal.goal_specific_information, NewHomeGoalInformation):
                info = goal.goal_specific_information
                goal_str += (f"\n  Location: {info.location}"
                            f"\n  Price: ${info.house_price:,.2f}"
                            f"\n  Deposit: ${info.deposit_amount:,.2f}"
                            f"\n  Target Date: {info.purchase_date}")
                            
            elif isinstance(goal.goal_specific_information, NewCarInformation):
                info = goal.goal_specific_information
                goal_str += (f"\n  Car: {info.car_type}"
                            f"\n  Price: ${info.car_price:,.2f}"
                            f"\n  Target Date: {info.purchase_date}")
                            
            elif isinstance(goal.goal_specific_information, OtherGoalInformation):
                info = goal.goal_specific_information
                goal_str += (f"\n  Description: {info.description}"
                            f"\n  Amount Needed: ${info.amount_required:,.2f}"
                            f"\n  Target Date: {info.target_date}")
            
            goals_info.append(goal_str)
        
        return f"{user_info}\n\nGoals:\n" + "\n\n".join(goals_info)

@dataclass
class ConversationState:
    finished: bool = False
    messages: list[Message] = field(default_factory=list)
    new_messages: list[Message] = field(default_factory=list)
    extracted_information: ExtractedInformation = field(default_factory=ExtractedInformation)

def parse_date(date_str: str) -> Optional[dt.date]:
    if not date_str:
        return None
    
    date_str = date_str.lower().strip()
    today = dt.date.today()
    
    # Handle relative dates like "in 3 months"
    if date_str.startswith("in "):
        try:
            num = int(re.search(r'\d+', date_str).group())
            if "day" in date_str:
                return today + dt.timedelta(days=num)
            elif "week" in date_str:
                return today + dt.timedelta(weeks=num)
            elif "month" in date_str:
                return today + dt.timedelta(days=30*num) # Rough approximation of 30 days per month
            elif "year" in date_str:
                return today + dt.timedelta(days=365*num) 
        except:
            pass
    
    # Handle relative dates like "next ..."
    if date_str == "next month":
        next_month = today.month + 1 if today.month < 12 else 1
        next_year = today.year if today.month < 12 else today.year + 1
        return dt.date(next_year, next_month, 1)
    
    if date_str == "next year":
        return dt.date(today.year + 1, 1, 1)
    
    # Try standard date formats
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"): # For loop through all the standards until 1 matches
        try:
            return dt.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    # Handle word format date
    if re.match(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}$', date_str, re.I):
        try:
            return dt.datetime.strptime(date_str, "%B %Y").date()
        except:
            try:
                return dt.datetime.strptime(date_str, "%b %Y").date()
            except:
                pass
    
    if re.match(r'^20\d{2}$', date_str):
        return dt.date(int(date_str), 1, 1)
    
    return None

def parse_currency(amount: str) -> Optional[float]:
    if not amount:
        return None
    
    amount = amount.lower().replace(",", "").strip()
    
    # Extract number
    try:
        numeric_part = float(re.search(r'[\d.]+', amount).group())
    except:
        return None
    
    # Extract amount and multiply to number
    if "million" in amount:
        return numeric_part * 1000000
    elif "billion" in amount:
        return numeric_part * 1000000000
    elif "thousand" in amount:
        return numeric_part * 1000
    elif "m" in amount and "million" not in amount:
        return numeric_part * 1000000
    elif "b" in amount and "billion" not in amount:
        return numeric_part * 1000000000
    elif "k" in amount:
        return numeric_part * 1000
    
    # Currently only keep amount, not regarding the currency
    if any(symbol in amount for symbol in "$£€¥"):
        return numeric_part
    
    return numeric_part

def extract_user_info(text: str) -> dict:
    # Use LLM to process natural language input from users
    prompt = f"""Extract personal information from this text: 
    "{text}"
    
    Rules:
    - First name and last name must be separated
    - Date of birth must be in YYYY-MM-DD format (convert if needed)
    - Email must be valid format
    
    Return JSON with these exact fields:
    {{
        "first_name": "string (required)",
        "last_name": "string (required)",
        "date_of_birth": "string as YYYY-MM-DD (required)",
        "email": "string (required)"
    }}
    """
    # use rules so that LLM output data in a way code can easily handle
    
    try:
        response = llm(prompt) # Call LLM API to process the text and extract structured information
        if not response:
            return {}
            
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        return json.loads(response.strip()) # Convert JSON string to Python dictionary
    except Exception as e:
        print(f"Error parsing user info: {e}")
        return {}

def extract_goal_details(text: str, goal_type: GoalType) -> Optional[Dict]:
    # Because we have specific goal that the user have, we can separate car and other goals easily. 
    if goal_type == GoalType.NEW_HOME:
        prompt = f"""Extract home purchase details from:
        "{text}"
        
        Return JSON with:
        - location
        - house_price (number)
        - deposit_amount (number)
        - purchase_date (YYYY-MM-DD format)
        
        Rules:
        - Convert amounts to numbers (e.g. "1.5M" → 1500000)
        - For relative dates like "in 3 years", calculate the exact date from today ({dt.date.today()})
        - For month names, use full month name (e.g. "January")
        """
    elif goal_type == GoalType.NEW_CAR:
        prompt = f"""Extract car purchase details from:
        "{text}"
        
        Return JSON with:
        - car_type (make + model)
        - car_price (number)
        - purchase_date (YYYY-MM-DD format)
        
        Rules:
        - Convert amounts to numbers (e.g. "20k" → 20000)
        - For relative dates like "next month", calculate the exact date from today ({dt.date.today()})
        """
    else:
        prompt = f"""Extract financial goal details from:
        "{text}"
        
        Return JSON with:
        - description
        - amount_required (number)
        - target_date (YYYY-MM-DD format)
        
        Rules:
        - Convert amounts to numbers
        - For relative dates like "by next year", calculate the exact date from today ({dt.date.today()})
        """
    
    try:
        response = llm(prompt) # Send user input to LLM to process
        if not response:
            return None
            
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        data = json.loads(response.strip())
        
        # Additional date validation
        if 'purchase_date' in data or 'target_date' in data:
            date_key = 'purchase_date' if 'purchase_date' in data else 'target_date'
            date_str = data[date_key]
            parsed_date = parse_date(date_str)
            if parsed_date:
                data[date_key] = str(parsed_date)
            else:
                return None
                
        return data
    except Exception as e:
        print(f"Error extracting goal details: {e}")
        return None

def chat_response(state: ConversationState) -> ConversationState:
    extracted = state.extracted_information
    last_message = state.messages[-1].text if state.messages else ""
    new_messages = []
    
    # If user wants to stop the conversation
    if last_message.lower().strip() in STOP_COMMANDS:
        extracted.conversation_stage = ConversationStage.COMPLETED
        new_messages.append(Message(
            text="Thank you for providing your financial goals information.",
            sender=Sender.AI
        ))
        return ConversationState(
            finished=True,
            messages=state.messages + state.new_messages,
            new_messages=new_messages,
            extracted_information=extracted
        )
    
    # Collect user info
    if extracted.conversation_stage == ConversationStage.GET_USER_INFO:
        user_info = extract_user_info(last_message)
        if all(k in user_info for k in ["first_name", "last_name", "date_of_birth", "email"]):
            extracted.user = User(
                first_name=user_info["first_name"],
                last_name=user_info["last_name"],
                date_of_birth=parse_date(user_info["date_of_birth"]),
                email=user_info["email"],
                goals=[]
            )
            extracted.conversation_stage = ConversationStage.GET_GOAL_TYPE
            new_messages.append(Message(
                text="Thank you! What financial goal would you like to discuss? (new home/new car/other)",
                sender=Sender.AI
            ))
        # Missed info then ask again, currently the flaw is it only recognize if all info is in the input
        else: # TO DO: add in handler that assess what is collected and form question based on what is missing
            new_messages.append(Message(
                text="Please provide your full name, date of birth (YYYY-MM-DD), and email",
                sender=Sender.AI
            ))
    
    # Get goal type
    elif extracted.conversation_stage == ConversationStage.GET_GOAL_TYPE:
        goal_type = last_message.lower()
        # Got goal type (home), send question to explain what detail is needed. 
        if "home" in goal_type:
            extracted.pending_goal = Goal(
                goal_type=GoalType.NEW_HOME,
                goal_name="Buy a new home",
                goal_specific_information=None
            )
            extracted.conversation_stage = ConversationStage.GET_GOAL_DETAILS
            new_messages.append(Message(
                text="Please tell me about the home you want to buy (location, price, deposit, and timeline)",
                sender=Sender.AI
            ))
        # Got goal type (car), send question to explain what detail is needed. 
        elif "car" in goal_type:
            extracted.pending_goal = Goal(
                goal_type=GoalType.NEW_CAR,
                goal_name="Buy a new car",
                goal_specific_information=None
            )
            extracted.conversation_stage = ConversationStage.GET_GOAL_DETAILS
            new_messages.append(Message(
                text="Please tell me about the car you want to buy (make/model, price, and timeline)",
                sender=Sender.AI
            ))
        # Got goal type (other), send question to explain what detail is needed. 
        else:
            extracted.pending_goal = Goal(
                goal_type=GoalType.OTHER,
                goal_name="Other financial goal",
                goal_specific_information=None
            )
            extracted.conversation_stage = ConversationStage.GET_GOAL_DETAILS
            new_messages.append(Message(
                text="Please describe your financial goal (what you want to achieve, how much money you'll need, and by when)",
                sender=Sender.AI
            ))
        # Ideally I want to be able to separate car and other more easily. But it is not able to
    
    # Goal specific information parsing
    elif extracted.conversation_stage == ConversationStage.GET_GOAL_DETAILS:
        goal_details = extract_goal_details(last_message, extracted.pending_goal.goal_type) # Use LLM to extract goals
        if goal_details:
            if extracted.pending_goal.goal_type == GoalType.NEW_HOME: # Since LLM returns different format, we can extract based on format for hom
                info = NewHomeGoalInformation(
                    location=goal_details.get("location", ""),
                    house_price=float(goal_details.get("house_price", 0)),
                    deposit_amount=float(goal_details.get("deposit_amount", 0)),
                    purchase_date=parse_date(goal_details.get("purchase_date", ""))
                )
                extracted.pending_goal.goal_specific_information = info
                extracted.conversation_stage = ConversationStage.CONFIRM_GOAL # Ask the user to confirm
                new_messages.append(Message(
                    text=f"Confirm: Buy home in {info.location} for ${info.house_price:,.2f} with ${info.deposit_amount:,.2f} deposit by {info.purchase_date}? (yes/no)",
                    sender=Sender.AI
                ))
                
            elif extracted.pending_goal.goal_type == GoalType.NEW_CAR: # Extract from car
                info = NewCarInformation(
                    car_type=goal_details.get("car_type", ""),
                    car_price=float(goal_details.get("car_price", 0)),
                    purchase_date=parse_date(goal_details.get("purchase_date", ""))
                )
                extracted.pending_goal.goal_specific_information = info
                extracted.conversation_stage = ConversationStage.CONFIRM_GOAL # Ask the user to confirm
                new_messages.append(Message(
                    text=f"Confirm: Buy {info.car_type} for ${info.car_price:,.2f} by {info.purchase_date}? (yes/no)",
                    sender=Sender.AI
                ))
                
            else: # Extract from other
                info = OtherGoalInformation(
                    description=goal_details.get("description", ""),
                    amount_required=float(goal_details.get("amount_required", 0)),
                    target_date=parse_date(goal_details.get("target_date", ""))
                )
                extracted.pending_goal.goal_specific_information = info
                extracted.conversation_stage = ConversationStage.CONFIRM_GOAL # Ask the user to confirm
                new_messages.append(Message(
                    text=f"Confirm: {info.description} requiring ${info.amount_required:,.2f} by {info.target_date}? (yes/no)",
                    sender=Sender.AI
                ))
        else: # LLM did not parse information, so ask again with clear example
            if extracted.pending_goal.goal_type == GoalType.NEW_HOME:
                new_messages.append(Message(
                    text="I couldn't understand those home details. Please provide: location, price, deposit, and timeline (e.g. 'Buy $1.5M home in London with 50k deposit in 3 years')",
                    sender=Sender.AI
                ))
            elif extracted.pending_goal.goal_type == GoalType.NEW_CAR:
                new_messages.append(Message(
                    text="I couldn't understand those car details. Please provide: make/model, price, and timeline (e.g. 'Buy a Tesla Model 3 for $40k in 6 months')",
                    sender=Sender.AI
                ))
            else:
                new_messages.append(Message(
                    text="I couldn't understand that goal. Please describe what you want to achieve, how much money you'll need, and by when (e.g. 'Start a business needing $50k by 2025')",
                    sender=Sender.AI
                ))
    
    # Check if user want to add more goals
    elif extracted.conversation_stage == ConversationStage.CONFIRM_GOAL:
        if last_message.lower().startswith("y"):
            extracted.user.goals.append(extracted.pending_goal)
            extracted.pending_goal = None
            extracted.conversation_stage = ConversationStage.ADD_ANOTHER_GOAL
            new_messages.append(Message(
                text="Goal saved! Would you like to add another goal? (yes/no)",
                sender=Sender.AI
            ))
        else:
            extracted.pending_goal = None
            extracted.conversation_stage = ConversationStage.GET_GOAL_TYPE
            new_messages.append(Message(
                text="Okay, let's try again. What goal would you like to discuss? (new home/new car/other)",
                sender=Sender.AI
            ))
    

    elif extracted.conversation_stage == ConversationStage.ADD_ANOTHER_GOAL:
        # they answered yes to adding more goals
        if last_message.lower().startswith("y"):
            extracted.conversation_stage = ConversationStage.GET_GOAL_TYPE
            new_messages.append(Message(
                text="What other financial goal would you like to discuss? (new home/new car/other)",
                sender=Sender.AI
            ))
        # they answered no to adding more goals
        else:
            extracted.conversation_stage = ConversationStage.COMPLETED
            new_messages.append(Message(
                text="Thank you for providing your financial goals information.",
                sender=Sender.AI
            ))
    
    return ConversationState(
        finished=extracted.conversation_stage == ConversationStage.COMPLETED,
        messages=state.messages + state.new_messages,
        new_messages=new_messages,
        extracted_information=extracted
    )