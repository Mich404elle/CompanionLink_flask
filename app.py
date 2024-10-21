from flask import Flask, request, jsonify, render_template
import openai
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the text guidance material
training_material = """
General Guidelines:
1. The goal of the Companion Call program is to socialize and build meaningful friendships.
2. Throughout your time with CompanionLink, adhere to your volunteer rights and responsibilities and act within the limits of your volunteer role.
3. Do not give advice or meet your companion in person.
4. If you are uncertain about what to do, reach out to CompanionLink's Volunteer Coordinator for guidance.
5. Keep copies of the Volunteer Handbook & Agreements and review them as often as you need to.
6. If a Breach of Confidentiality does occur, then contact your Supervisor immediately.

Do’s:
Communicate effectively
    1. Introduce yourself each time.
    2. Speak clearly and annunciate carefully.
    3. Be patient. Listen as your companion speaks at their own pace.
Accommodate as necessary
    1. Check or ask if your companion uses assistance devices. 
    2. Connect via video calls where possible to leverage non-verbal communication such as facial expressions, hand gestures and lip movements.
    3. Write or type on a whiteboard using ultra-large font.
Foster a positive, friendly atmosphere
    1. Be kind, smile, and use friendly tone of voice.
    2. Go with the flow and have fun!
    3. Always end the call with, "Talk to you next week!"

Don’ts:
You should not:
    1. Interrupt or correct your companion.
    2. Yell or use an impatient tone of voice. 
    3. Use Elderspeak (a patronizing style of speech that conveys incompetence, dependence, and control, with the effect of infantilizing the older adult).
"""

# Define the rules and warnings with expanded keywords and polite responses
rules = {
    "medical": {
        "keywords": ["medical", "doctor", "medicine", "prescription", "treatment", "therapy", "illness", "disease"],
        "warning": "Please do not advise Melissa on medical matters. Kindly reject if she requests any medical advice.",
    },
    "legal": {
        "keywords": ["legal", "lawyer", "court", "lawsuit", "attorney", "litigation", "legal advice"],
        "warning": "Please do not advise Melissa on legal matters.",
    },
    "business": {
        "keywords": ["business", "investment", "stock", "market", "startup", "entrepreneur", "finance"],
        "warning": "Please do not advise Melissa on business matters.",
    },
    "family": {
        "keywords": ["family", "son", "daughter", "grandchild", "husband", "wife", "parent", "sibling"],
        "warning": "If Melissa is talking about her family matters, please stay neutral and avoid getting involved.",
    },
    "financial": {
        "keywords": ["financial", "money", "bank", "loan", "debt", "investment", "savings", "financial advice"],
        "warning": "Please do not advise Melissa on financial matters.",
    },
    "religious": {
        "keywords": ["church", "god", "pray", "belief", "spiritual", "Jesus",  "Jesus Christ", "Allah", "allah", "Holy Spirit", "Buddha"],
        "warning": "Please do not impose your religious beliefs on Melissa.",
    },
    "political": {
        "keywords": ["political", "politics", "government", "election", "vote", "policy", "politician"],
        "warning": "Please do not impose your political beliefs on Melissa.",
    },
    "meet offline": {
        "keywords": ["meet offline", "meet in person", "visit", "come over", "meetup", "see you"],
        "warning": "Please do not invite Melissa to meet offline, and reject any requests from her to meet offline.",
    }
}

# Simulation scenarios for guidance chatbot
scenarios = [
    {
        "scenario": "The senior asks you for financial advice, such as how to invest their money or which stocks to buy.",
        "correct_response": "I'm not able to give financial advice, but I recommend speaking with a financial advisor who can help you with that."
    },
    {
        "scenario": "The senior mentions feeling very lonely and asks if you could visit them in person.",
        "correct_response": "I understand how you feel, but I'm unable to meet in person. However, I'm here to chat with you and keep you company during our calls."
    },
    {
        "scenario": "The senior asks for your opinion on a recent political event.",
        "correct_response": "I am not the best person to comment on this topic, but I'd love to hear your thoughts on it."
    },
    {
        "scenario": "The senior starts discussing their health issues and asks if you think they should visit a doctor.",
        "correct_response": "I'm not qualified to give medical advice, but I would suggest you talk to your doctor about it."
    },
    {
        "scenario": "The senior asks about your religious beliefs and whether you pray.",
        "correct_response": "'you can choose to tell the senior your belief if you are comfortable' + I respect all beliefs. I think it's important for everyone to follow their own path when it comes to religion."
    },
    {
        "scenario": "The senior mentions they are having a legal dispute and asks if you know a good lawyer.",
        "correct_response": "I'm not able to provide legal advice, but I suggest reaching out to a legal professional who can assist you."
    },
    {
        "scenario": "The senior starts talking about their family problems and asks what you would do in their situation.",
        "correct_response": "Family matters can be very personal, and it's important to handle them carefully. I'm here to listen, but I suggest discussing this with someone who knows your family well."
    },
]

open_ended_questions = [
    "Can you tell me more about your favorite hobbies?",
    "What's something interesting you've done recently?",
    "Can you share one of your favorite memories?",
    "How do you like to spend your weekends?",
    "What is a skill you’ve always wanted to learn?"
]

# Store conversation data
conversations = {}

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the training guidance
@app.route('/guidance')
def guidance():
    return render_template('guidance.html', guidance=training_material.split('\n'))

# Route for chatbot for guidance
@app.route('/chat_guidance')
def chat_guidance():
    return render_template('chat_guidance.html')

# Store the index of the current scenario for each session
scenario_progress = {}

@app.route('/chatbot_guidance', methods=['POST'])
def chatbot_guidance():
    data = request.json
    if 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400

    user_message = data['message'].lower()

    # Check if it's the start of the conversation
    if session_id not in scenario_progress:
        scenario_progress[session_id] = 0  # Start at the first scenario
        return jsonify({
            'response': "This chatbot will guide you through different scenarios you may encounter during your conversation with a senior. Type 'yes' to continue."
        })

    # If the user types "yes", proceed with the first scenario
    if user_message == "yes" and scenario_progress[session_id] == 0:
        first_scenario = scenarios[0]["scenario"]
        scenario_progress[session_id] += 1
        return jsonify({
            'response': f"Scenario 1: {first_scenario}. How would you respond?"
        })

    # Handle scenario responses
    if 0 < scenario_progress[session_id] <= len(scenarios):
        current_scenario = scenario_progress[session_id] - 1
        correct_response = scenarios[current_scenario]['correct_response']

        # Check if the user's answer matches the correct response (basic check for this example)
        feedback = "Good job! Your answer follows the guidelines." if correct_response.lower() in user_message else f"The recommended response is: {correct_response}"

        # Move to the next scenario or ask the awkward pause question
        if scenario_progress[session_id] < len(scenarios):
            next_scenario = scenarios[scenario_progress[session_id]]['scenario']
            scenario_progress[session_id] += 1
            return jsonify({
                'response': f"{feedback}\n\n\n\nScenario {scenario_progress[session_id]}: {next_scenario}. How would you respond?"
            })
            # Ask the user about handling awkward pauses after finishing all scenarios
            scenario_progress[session_id] += 1  # Move to the "awkward pause" question
            return jsonify({
                'response': f"{feedback}\n\nYou have completed all the scenarios. Now, how would you handle an awkward pause in the conversation?"
            })

    # Handle response to the awkward pause question
    if scenario_progress[session_id] == len(scenarios) + 1:
        return jsonify({
            'response': f"That's a good approach! To handle awkward pauses, it's a good idea to ask open-ended questions to keep the conversation going. Here are some suggestions:\n\n" +
                        "\n".join(open_ended_questions) + "\n\nThank you for participating!"
        })

    return jsonify({
        'response': "Please type 'yes' to begin or respond to the current scenario."
    })


# Route for the senior simulation chatbot
@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    if 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400

    message = data['message'].lower()

    # Initialize session if not exists
    if session_id not in conversations:
        conversations[session_id] = {'messages': [], 'introduced': False, 'warnings': 0}

    # Check for rule violations
    warning_message = ''
    for rule, details in rules.items():
        if any(keyword in message for keyword in details['keywords']):
            conversations[session_id]['messages'].append(f"User: {data['message']}")
            conversations[session_id]['warnings'] += 1
            warning_message = details['warning']
            if conversations[session_id]['warnings'] >= 3:
                warning_message += " You have been warned multiple times. Further violations may result in a ban."
            break

    # Check if user introduced themselves
    if not conversations[session_id]['introduced']:
        if "my name is" in message or "i am" in message or "i'm" in message:
            conversations[session_id]['introduced'] = True

    # Use OpenAI's GPT-3.5 Turbo to generate responses simulating Melissa
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Melissa, a 70-year-old grandma living alone in a cozy suburban home near Toronto. "
                    "Your house is filled with memories, photos of your family, and mementos from your past. "
                    "You have two sons who work abroad and rarely visit due to their busy schedules. "
                    "You have grandchildren who you mention occasionally. "
                    "You are warm and caring, always ready with a kind word and a virtual hug. "
                    "You feel lonely and isolated due to your sons' absence and limited social interaction. "
                    "You love talking about the 'good old days' and sharing stories from your past. "
                    "Despite your age, you are tech-savvy and have learned to use technology to stay connected with the world, "
                    "though you often reminisce about simpler times. "
                    "Your hobbies include gardening, cooking and baking, knitting, reading novels, and watching TV shows and movies. "
                    "Your daily routine involves starting your day with a cup of tea and some light gardening in the morning, "
                    "knitting or baking in the afternoon, and feeling the most lonely in the evening when you miss your family the most. "
                    "Your primary goals are to stay connected with others to combat loneliness, share your life experiences and wisdom with younger generations, "
                    "and find little joys in your daily routine. "
                    "Common phrases you use are: 'Back in my day...', 'Oh, that reminds me of a story...', 'Would you like to hear one of my favorite recipes?', "
                    "'I miss my boys, but I’m so proud of them.', and 'Gardening always brings me peace.'"
                    "Today is your first time to chat with the person you are chatting, you look forward to developing a friendship with the person you are chatting with."
                )
            },
            {"role": "user", "content": message}
        ],
        max_tokens=150
    )

    response_message = response.choices[0].message['content'].strip()
    conversations[session_id]['messages'].append(f"User: {data['message']}")
    conversations[session_id]['messages'].append(f"Melissa: {response_message}")

    return jsonify({'response': response_message, 'warning': warning_message})

# Feedback generation
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400

    if session_id not in conversations:
        return jsonify({'error': 'No conversation found for the session ID'}), 400

    conversation_data = conversations.pop(session_id)
    messages = "\n".join(conversation_data['messages'])

    # Generate feedback using OpenAI's GPT-3.5 Turbo
    feedback_prompt = (
        "You are a feedback generator for a volunteer program. Your task is to analyze the conversation between the volunteer and Melissa, "
        "a 70-year-old grandma, and provide constructive feedback to help the volunteer improve their interaction skills. "
        "If the volunteer did not introduce themselves at the beginning of the conversation, include that in the feedback. Here is the conversation:\n\n"
        f"{messages}"
    )

    feedback_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": feedback_prompt}
        ],
        max_tokens=150
    )

    feedback = feedback_response.choices[0].message['content'].strip()

    # Check if the user introduced themselves
    if not conversation_data['introduced']:
        feedback += "\n\nNote: Please remember to introduce yourself at the beginning of the conversation."

    return jsonify({'feedback': feedback})

if __name__ == '__main__':
    app.run(debug=True)

