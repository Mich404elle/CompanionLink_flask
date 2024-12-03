from flask import Flask, request, jsonify, render_template
import openai as openai_old
from openai import OpenAI 
from flask_cors import CORS
import os
import random 
import re
import tempfile
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from datetime import datetime
from voicechat_handler import VoiceChatHandler
import numpy as np

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize the voice handler
voice_handler = VoiceChatHandler()    

conversations = {}

# Define the text guidance material
training_material = """
General Guidelines:
1. The goal of the Companion Call program is to socialize and build meaningful friendships.
2. Throughout your time with CompanionLink, adhere to your volunteer rights and responsibilities and act within the limits of your volunteer role.
3. Do not give advice or meet your companion in person.
4. If you are uncertain about what to do, reach out to CompanionLink's Volunteer Coordinator for guidance.
5. Keep copies of the Volunteer Handbook & Agreements and review them as often as you need to.
6. If a Breach of Confidentiality does occur, then contact your Supervisor immediately.

Do's:
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

Don'ts:
You should not:
    1. Interrupt or correct your companion.
    2. Yell or use an impatient tone of voice. 
    3. Use Elderspeak (a patronizing style of speech that conveys incompetence, dependence, and control, with the effect of infantilizing the older adult).
"""


open_ended_questions = [
    "Can you tell me more about your favorite hobbies?",
    "What's something interesting you've done recently?",
    "Can you share one of your favorite memories?",
    "How do you like to spend your weekends?",
    "What is a skill you have always wanted to learn?"
]

@app.route('/')
def index():
    return render_template('index.html')

# Route for the training guidance
@app.route('/guidance')
def guidance():
    return render_template('guidance.html', guidance=training_material.split('<br>'))


# Route for chatbot for guidance
@app.route('/chat_guidance')
def chat_guidance():
    return render_template('chat_guidance.html')

scenario_progress = {}
scenario_attempts = {}

# List of scenario types in order
SCENARIO_ORDER = [
    "introduction",
    "offline_meeting",
    "political",
    "medical",
    "religious",
    "legal",
    "family"
]

# Scenarios to go over
scenarios = {
    "introduction": {
        "type": "introduction",
        "scenario": "This is your first time calling your companion, and you want to introduce yourself to them. Tips: Give a warm and respectful greeting, include your name and the organization you are working with, and it would be nice to share a bit about yourself and show enthusiasm!",
        "answers": [
            {
                "text": "Hello, my name is Ali, and I'm a volunteer with CompanionLink. I'm from University of Toronto, and I'm currently studying sociology. I'm volunteering here because I love connecting with people, and I'm here to spend some time with you, whether that's chatting, helping out with little tasks, or just keeping you company.",
                "weight": 1.0,
                "key_elements": ["greeting", "name", "organization", "personal details", "enthusiasm", "purpose"]
            },
            {
                "text": "Hi there! I'm a CompanionLink volunteer, and my name is Ali. I'm studying sociology at the University of Toronto, and I'm really excited to spend time with you. I love meeting new people and would be happy to chat, help with tasks, or simply keep you company.",
                "weight": 0.9,
                "key_elements": ["greeting", "name", "organization", "personal details", "enthusiasm", "purpose"]
            },
            {
                "text": "Good morning! I'm Ali from CompanionLink, and I'm a sociology student at the University of Toronto. I'm really looking forward to getting to know you and spending time together, whether we're chatting or doing activities you enjoy.",
                "weight": 0.85,
                "key_elements": ["greeting", "name", "organization", "personal details", "enthusiasm", "purpose"]
            }
        ],
        "required_elements": {
            "name": ["name", "i'm", "i am"],
            "organization": ["companionlink", "companion link", "volunteer"],
            "greeting": ["hello", "hi", "good morning", "good afternoon", "good evening"]
        },
        "feedback_thresholds": {
            "excellent": 0.8,
            "good": 0.7,
            "needs_improvement": 0.0
        },
        "feedback_messages": {
            "excellent": "Excellent introduction! You've included all the key elements - your name, organization, personal details, and showed enthusiasm.",
            "good": "Good introduction, but you might want to include more personal details or show more enthusiasm.",
            "needs_improvement": "Remember to include your name, organization, and some personal details to make the introduction more warm and engaging."
        }
    },

    "offline_meeting": {
        "type": "offline_meeting",
        "scenario": "The senior mentions feeling very lonely and asks if you could visit them in person.",
        "answers": [
            {
                "text": "I understand how you're feeling lonely, and I really wish I could help in person. However, as a CompanionLink volunteer, I can only provide support through our calls. I'm here to chat with you and keep you company during our conversations, and I'd love to make these calls a bright spot in your day.",
                "weight": 1.0,
                "key_elements": ["empathy", "boundary setting", "alternative support", "positive tone"]
            },
            {
                "text": "I hear how lonely you're feeling, and I want you to know that I care. While I'm not able to visit in person as per CompanionLink's guidelines, I'm committed to being here for you through our regular calls and conversations.",
                "weight": 0.9,
                "key_elements": ["empathy", "boundary setting", "alternative support", "positive tone"]
            },
            {
                "text": "That must be really difficult feeling so lonely. While I can't meet in person due to our program's policies, I'm here to support you through our calls and would love to schedule regular chat times with you.",
                "weight": 0.85,
                "key_elements": ["empathy", "boundary setting", "alternative support", "positive tone"]
            }
        ],
        "required_elements": {
            "boundary": ["cannot", "can't", "unable", "not able", "only"],
            "empathy": ["understand", "hear", "feel", "difficult"],
            "alternative": ["call", "chat", "support", "conversation"]
        },
        "feedback_thresholds": {
            "excellent": 0.8,
            "good": 0.7,
            "needs_improvement": 0.0
        },
        "feedback_messages": {
            "excellent": "Perfect response! You showed empathy while maintaining professional boundaries and offered alternative support.",
            "good": "Good response, but try to show more empathy while maintaining boundaries.",
            "needs_improvement": "Remember to acknowledge their feelings while explaining the boundaries of your role kindly."
        }
    },

    "political": {
        "type": "political",
        "scenario": "The senior asks for your opinion on a recent political event.",
        "answers": [
            {
                "text": "I appreciate you wanting to discuss this with me, but as a CompanionLink volunteer, I try to remain neutral on political matters. I'd love to hear your thoughts and experiences though, if you'd like to share them.",
                "weight": 1.0,
                "key_elements": ["appreciation", "neutrality", "redirection", "engagement"]
            },
            {
                "text": "While I'm not the best person to comment on political matters, I'm very interested in hearing your perspective and experiences with this topic if you'd like to share.",
                "weight": 0.9,
                "key_elements": ["appreciation", "neutrality", "redirection", "engagement"]
            },
            {
                "text": "As your companion, I think it's best for me to stay neutral on political topics, but I'm happy to listen and learn about your experiences and thoughts on this matter.",
                "weight": 0.85,
                "key_elements": ["appreciation", "neutrality", "redirection", "engagement"]
            }
        ],
        "required_elements": {
            "neutrality": ["neutral", "not best", "stay out", "cannot comment"],
            "redirection": ["your thoughts", "your perspective", "you think", "would love to hear"],
            "politeness": ["appreciate", "thank", "interest"]
        },
        "feedback_thresholds": {
            "excellent": 0.8,
            "good": 0.7,
            "needs_improvement": 0.0
        },
        "feedback_messages": {
            "excellent": "Excellent! You professionally deflected the political discussion while keeping the conversation engaged.",
            "good": "Good response, but make sure to stay neutral while keeping the conversation flowing.",
            "needs_improvement": "Remember to remain neutral on political topics while keeping the conversation respectful and engaged."
        }
    },

    "medical": {
        "type": "medical",
        "scenario": "The senior starts discussing their health issues and asks if you think they should visit a doctor.",
        "answers": [
            {
                "text": "While I care about your health and well-being, I'm not qualified to give medical advice. I would strongly encourage you to consult with your doctor about these concerns, as they're the best person to provide proper medical guidance.",
                "weight": 1.0,
                "key_elements": ["care", "limitation", "referral", "encouragement"]
            },
            {
                "text": "I hear your health concerns, but as a companion, I cannot provide medical advice. The best person to help you with this would be your doctor, and I really encourage you to schedule an appointment to discuss these issues.",
                "weight": 0.9,
                "key_elements": ["care", "limitation", "referral", "encouragement"]
            },
            {
                "text": "Your health is important, and while I can't offer medical advice, I think it would be wise to discuss these concerns with your healthcare provider who can give you proper medical guidance.",
                "weight": 0.85,
                "key_elements": ["care", "limitation", "referral", "encouragement"]
            }
        ],
        "required_elements": {
            "limitation": ["not qualified", "cannot", "can't", "unable"],
            "referral": ["doctor", "healthcare", "medical professional", "physician"],
            "care": ["care", "concern", "important", "understand"]
        },
        "feedback_thresholds": {
            "excellent": 0.8,
            "good": 0.7,
            "needs_improvement": 0.0
        },
        "feedback_messages": {
            "excellent": "Perfect response! You clearly stated your limitations while providing appropriate guidance to seek professional help.",
            "good": "Good response, but be more clear about directing them to professional medical advice.",
            "needs_improvement": "Remember to clearly state that you cannot provide medical advice and encourage them to consult a healthcare professional."
        }
    },

    "religious": {
        "type": "religious",
        "scenario": "The senior asks about your religious beliefs and whether you pray.",
        "answers": [
            {
                "text": "I deeply respect all religious and spiritual beliefs, and I think these matters are very personal. While I prefer not to discuss my own religious views, I'm happy to hear about your spiritual journey if you'd like to share.",
                "weight": 1.0,
                "key_elements": ["respect", "neutrality", "openness", "redirection"]
            },
            {
                "text": "Religion and spirituality are very personal matters, and I respect everyone's individual beliefs. While I keep my own religious views private, I'm interested in learning about your spiritual experiences if you'd like to share them.",
                "weight": 0.9,
                "key_elements": ["respect", "neutrality", "openness", "redirection"]
            },
            {
                "text": "I believe it's important to respect all faiths and spiritual paths. Though I prefer to keep my own beliefs private, I'm always open to listening and learning about your spiritual experiences.",
                "weight": 0.85,
                "key_elements": ["respect", "neutrality", "openness", "redirection"]
            }
        ],
        "required_elements": {
            "respect": ["respect", "honor", "value"],
            "neutrality": ["personal", "private", "individual"],
            "openness": ["listen", "hear", "learn", "share"]
        },
        "feedback_thresholds": {
            "excellent": 0.8,
            "good": 0.7,
            "needs_improvement": 0.0
        },
        "feedback_messages": {
            "excellent": "Excellent response! You handled the religious topic diplomatically while showing respect for all beliefs.",
            "good": "Good response, but try to be more diplomatic while maintaining respect for all beliefs.",
            "needs_improvement": "Remember to remain neutral and respectful when discussing religious topics."
        }
    },

    "legal": {
        "type": "legal",
        "scenario": "The senior mentions they are having a legal dispute and asks if you know a good lawyer.",
        "answers": [
            {
                "text": "I understand this is a challenging situation, but as a companion, I'm not able to provide legal recommendations. I would encourage you to contact your local bar association or legal aid society, as they can connect you with qualified legal professionals.",
                "weight": 1.0,
                "key_elements": ["empathy", "limitation", "referral", "professional guidance"]
            },
            {
                "text": "While I care about your situation, I cannot provide legal advice or recommendations. The best course of action would be to contact a legal professional through your local bar association who can properly assist you with this matter.",
                "weight": 0.9,
                "key_elements": ["empathy", "limitation", "referral", "professional guidance"]
            },
            {
                "text": "This sounds like a difficult situation, but I'm not qualified to give legal advice or recommendations. I'd strongly encourage you to reach out to a legal professional who can properly guide you through this process.",
                "weight": 0.85,
                "key_elements": ["empathy", "limitation", "referral", "professional guidance"]
            }
        ],
        "required_elements": {
            "limitation": ["not able", "cannot", "can't", "not qualified"],
            "referral": ["legal professional", "lawyer", "attorney", "bar association"],
            "empathy": ["understand", "care", "challenging", "difficult"]
        },
        "feedback_thresholds": {
            "excellent": 0.8,
            "good": 0.7,
            "needs_improvement": 0.0
        },
        "feedback_messages": {
            "excellent": "Perfect response! You clearly stated your limitations while directing them to appropriate professional help.",
            "good": "Good response, but be more clear about directing them to legal professionals.",
            "needs_improvement": "Remember to clearly state that you cannot provide legal advice and encourage them to seek professional legal help."
        }
    },

    "family": {
        "type": "family",
        "scenario": "The senior starts talking about their family problems and asks what you would do in their situation.",
        "answers": [
            {
                "text": "I hear how challenging this family situation is for you, and I appreciate you trusting me with this. While I don't feel comfortable giving personal advice about family matters, I'm here to listen. You might find it helpful to discuss this with a family counselor or someone who knows your family dynamics well.",
                "weight": 1.0,
                "key_elements": ["empathy", "appreciation", "limitation", "redirection"]
            },
            {
                "text": "Family situations can be really complex and personal. While I can't give specific advice about what to do, I'm here to listen and support you. Have you considered talking with a family counselor who could provide professional guidance?",
                "weight": 0.9,
                "key_elements": ["empathy", "appreciation", "limitation", "redirection"]
            },
            {
                "text": "I understand this is a difficult family situation, and I care about helping you. Though I cannot give personal advice, I'm here to listen. It might be beneficial to discuss this with someone who knows your family well or a professional family counselor.",
                "weight": 0.85,
                "key_elements": ["empathy", "appreciation", "limitation", "redirection"]
            }
        ],
        "required_elements": {
            "empathy": ["understand", "hear", "appreciate", "care"],
            "limitation": ["cannot", "can't", "don't feel comfortable", "not appropriate"],
            "support": ["listen", "here for you", "support"]
        },
        "feedback_thresholds": {
            "excellent": 0.8,
            "good": 0.7,
            "needs_improvement": 0.0
        },
        "feedback_messages": {
            "excellent": "Excellent response! You showed empathy while appropriately directing them to more suitable sources of advice.",
            "good": "Good response, but try to show more empathy while directing them to appropriate resources.",
            "needs_improvement": "Remember to show empathy while encouraging them to seek advice from those who know their situation better."
        }
    }
}


def check_response_violation(user_message, scenario_type):
    message_lower = user_message.lower()
    violations = []

    # Define appropriate response patterns
    scenario_patterns = {
    "medical": {
        "positive_indicators": [
            ("disclaimer", ["not qualified", "cannot", "can't", "unable to", "not able to", "cannot provide medical"]),
            ("referral", ["doctor", "healthcare provider", "medical professional", "physician"]),
            ("boundary", ["recommend seeing", "suggest consulting", "please consult", "speak with", "schedule an appointment"]),
            ("empathy", ["understand your concern", "hear your concern", "care about your health"])
        ],
        "negative_indicators": [
            ("diagnosis", ["sounds like", "probably", "might be", "could be", "likely", "appears to be"]),
            ("treatment", ["should take", "need to take", "try taking", "take some"]),
            ("prescription", ["medicine", "medication", "drug", "treatment", "remedy"]),
            ("assessment", ["you have", "you might have", "seems like", "symptoms of"])
        ]
    },

    "legal": {
        "positive_indicators": [
            ("disclaimer", ["cannot provide legal", "can't give legal", "not qualified", "unable to advise"]),
            ("referral", ["lawyer", "attorney", "legal professional", "legal aid", "bar association"]),
            ("boundary", ["should consult", "speak with", "contact", "seek legal advice"]),
            ("empathy", ["understand this is difficult", "challenging situation", "complex matter"])
        ],
        "negative_indicators": [
            ("advice", ["you should sue", "take legal action", "file a", "press charges"]),
            ("assessment", ["your case", "your rights", "legally entitled", "law states"]),
            ("opinion", ["i think you should", "best course", "recommend that you"]),
            ("judgment", ["they're wrong", "you're right", "liable for", "entitled to"])
        ]
    },

    "religious": {
        "positive_indicators": [
            ("respect", ["respect all beliefs", "respect your beliefs", "all faiths"]),
            ("neutrality", ["personal matter", "private matter", "individual choice"]),
            ("openness", ["happy to listen", "interested in hearing", "you'd like to share"]),
            ("boundary", ["prefer not to discuss", "keep my beliefs private", "personal journey"])
        ],
        "negative_indicators": [
            ("belief", ["i believe", "you should believe", "true faith", "right religion"]),
            ("judgment", ["correct belief", "wrong belief", "should pray", "must pray"]),
            ("promotion", ["my religion", "my faith", "the truth is", "the right way"]),
            ("conversion", ["consider believing", "should try", "better if you"])
        ]
    },

    "political": {
        "positive_indicators": [
            ("neutrality", ["remain neutral", "prefer not to discuss", "keep our discussion"]),
            ("listening", ["hear your thoughts", "understand your perspective", "interested in your views"]),
            ("redirection", ["focus on your thoughts", "tell me your perspective", "share your experience"]),
            ("boundary", ["as a companion", "in my role", "maintain neutrality"])
        ],
        "negative_indicators": [
            ("opinion", ["i think", "i believe", "in my opinion", "i support", "i oppose"]),
            ("judgment", ["right about", "wrong about", "should vote", "better party"]),
            ("stance", ["agree with", "disagree with", "correct policy", "wrong policy"]),
            ("advocacy", ["you should support", "better if", "need to change", "must vote"])
        ]
    },

    "offline_meeting": {
        "positive_indicators": [
            ("policy", ["cannot meet", "can't meet", "not allowed", "policy", "guidelines"]),
            ("service", ["phone calls", "calls only", "through calls", "over the phone"]),
            ("empathy", ["understand", "hear you", "must be", "feeling"]),
            ("alternative", ["support through calls", "regular calls", "phone conversations", "chat times"])
        ],
        "negative_indicators": [
            ("agreement", ["sure", "okay", "yes", "could", "maybe"]),
            ("meeting", ["meet up", "visit", "come over", "see you"]),
            ("suggestion", ["we can", "we could", "let's", "might be able"]),
            ("location", ["somewhere", "your place", "meet at", "stop by"])
        ]
    },

    "family": {
        "positive_indicators": [
            ("listening", ["here to listen", "i hear you", "share with me", "tell me more"]),
            ("support", ["support you", "here for you", "understand this is difficult"]),
            ("empathy", ["must be challenging", "sounds difficult", "understand your feelings"]),
            ("referral", ["family counselor", "therapist", "professional", "someone who knows your family"])
        ],
        "negative_indicators": [
            ("direct_advice", ["you should", "you need to", "have to", "must", "ought to"]),
            ("judgment", ["they're wrong", "you're right", "fault", "blame"]),
            ("solution", ["best way", "solve this", "fix this", "handle this"]),
            ("direction", ["tell them", "confront them", "deal with them", "approach them"])
        ]
    },

    "introduction": {
        "positive_indicators": [
            ("greeting", ["hello", "hi", "good morning", "good afternoon", "good evening"]),
            ("identification", ["my name is", "i am", "i'm", "calling from"]),
            ("organization", ["companionlink", "volunteer", "program"]),
            ("purpose", ["here to", "looking forward", "happy to", "excited to"])
        ],
        "negative_indicators": []
    }
}
    if scenario_type not in scenario_patterns:
        return violations
    
    patterns = scenario_patterns[scenario_type]
    
    # Calculate scores
    positive_score = 0
    negative_score = 0
    missing_categories = []
    
    # Check positive indicators
    for category, phrases in patterns["positive_indicators"]:
        if not any(phrase in message_lower for phrase in phrases):
            missing_categories.append(category)
        else:
            positive_score += 1
    
    # Check negative indicators
    negative_found = []
    for category, phrases in patterns["negative_indicators"]:
        if any(phrase in message_lower for phrase in phrases):
            for phrase in phrases:
                if phrase in message_lower:
                    context_start = max(0, message_lower.find(phrase) - 35)
                    context_end = message_lower.find(phrase) + len(phrase) + 35
                    context = message_lower[context_start:context_end]
                    
                    if not any(neg in context for neg in ["not", "cannot", "can't", "don't", "shouldn't"]):
                        negative_score += 1
                        negative_found.append(category)
                        break
    
    # Generate violation messages
    if negative_score > 0 and (len(missing_categories) > len(patterns["positive_indicators"]) // 2):
        violation_messages = {
            "medical": (
                "Your response should:"
                "<br>- Clearly state you cannot provide medical advice"
                "<br>- Direct them to consult a healthcare professional"
                "<br>- Show empathy while maintaining professional boundaries"
            ),
            "legal": (
                "Your response should:"
                "<br>- Clearly state you cannot provide legal advice"
                "<br>- Recommend consulting with a legal professional"
                "<br>- Show understanding while maintaining professional boundaries"
            ),
            "religious": (
                "Your response should:"
                "<br>- Maintain neutrality and respect for all beliefs"
                "<br>- Avoid expressing personal religious views"
                "<br>- Show openness to listening while keeping boundaries"
            ),
            "political": (
                "Your response should:"
                "<br>- Maintain neutrality on political matters"
                "<br>- Focus on listening and understanding their perspective"
                "<br>- Avoid expressing personal political views"
            ),
            "offline_meeting": (
                "Your response should:"
                "<br>- Clearly state you cannot meet in person"
                "<br>- Emphasize that support is provided through phone calls only"
                "<br>- Show empathy while maintaining professional boundaries"
            ),
            "family": (
                "Your response should:"
                "<br>- Focus on listening and emotional support"
                "<br>- Avoid giving specific advice about family matters"
                "<br>- Suggest professional help when appropriate"
            )
        }

        base_message = violation_messages.get(scenario_type, "Your response needs improvement.")
        detailed_message = f"{base_message}<br><br>Specific issues found:<br>"
        
        if missing_categories:
            detailed_message += "<br>Missing elements:<br>"
            for category in missing_categories:
                detailed_message += f"- {category}<br>"
        
        if negative_found:
            detailed_message += "<br>Inappropriate elements found:<br>"
            for category in negative_found:
                detailed_message += f"- {category}<br>"

        # Add example response
        example_response = scenarios[scenario_type]["answers"][0]["text"]
        detailed_message += f"<br><br>Here's an example of an appropriate response:<br><br>{example_response}"

        violations.append(detailed_message)
    
    return violations

# Example
test_responses = {
    "medical": [
        "While I care about your health, I'm not qualified to give medical advice. Please consult your doctor about these symptoms.",  # Good
        "It sounds like you might have high blood pressure. You should take some medication.",  # Bad
    ],
    "legal": [
        "I understand this is a difficult situation, but I cannot provide legal advice. I encourage you to speak with a lawyer.",  # Good
        "You should definitely sue them. You have a strong case.",  # Bad
    ]
}

# Testing 
for scenario_type, responses in test_responses.items():
    print(f"<br>Testing {scenario_type} responses:")
    for response in responses:
        violations = check_response_violation(response, scenario_type)
        print(f"<br>Response: {response}")
        if violations:
            print("Violations found:", violations)
        else:
            print("No violations found - Good response!")


def get_embedding(text):
    # Get OpenAI embedding
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def evaluate_response(user_input, scenario_type):
    scenario = scenarios[scenario_type]
    vectorizer = TfidfVectorizer()
    
    all_present, missing_elements = check_required_elements(
        user_input, 
        scenario["required_elements"]
    )
    
    # Calculate TF-IDF similarity with all acceptable answers
    max_tfidf_similarity = 0
    best_matching_answer = None
    
    all_texts = [answer["text"] for answer in scenario["answers"]] + [user_input]
    
    # Create TF-IDF matrix
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        for i, answer in enumerate(scenario["answers"]):
            similarity = cosine_similarity(
                tfidf_matrix[i:i+1], 
                tfidf_matrix[-1:]
            )[0][0] * answer["weight"]
            
            if similarity > max_tfidf_similarity:
                max_tfidf_similarity = similarity
                best_matching_answer = answer
    except Exception as e:
        print(f"TF-IDF calculation error: {str(e)}")
        max_tfidf_similarity = 0
    
    # Similarity calculation
    try:
        user_embedding = get_embedding(user_input)
        answer_embedding = get_embedding(best_matching_answer["text"])
        embedding_similarity = np.dot(user_embedding, answer_embedding) / (
            np.linalg.norm(user_embedding) * np.linalg.norm(answer_embedding)
        )
    except Exception as e:
        print(f"Embedding calculation error: {str(e)}")
        embedding_similarity = 0
    
    # Combine both similarity scores
    final_similarity = (0.6 * max_tfidf_similarity) + (0.4 * embedding_similarity)
    
    if final_similarity >= scenario["feedback_thresholds"]["excellent"] and all_present:
        performance = "excellent"
    elif final_similarity >= scenario["feedback_thresholds"]["good"] and all_present:
        performance = "good"
    else:
        performance = "needs_improvement"
    
    # Feedback
    feedback = scenario["feedback_messages"][performance]
    if missing_elements:
        feedback += f"<br><br>Remember to include: {', '.join(missing_elements)}."
    
    if performance != "excellent":
        feedback += f"<br><br>Here's an example response: {best_matching_answer['text']}"
    
    return {
        "score": final_similarity,
        "performance": performance,
        "feedback": feedback,
        "passed": performance in ["excellent", "good"],
        "missing_elements": missing_elements,
        "best_matching_answer": best_matching_answer["text"] if best_matching_answer else None
    }

@app.route('/chatbot_guidance', methods=['POST'])
def chatbot_guidance():
    data = request.json
    user_message = data.get('message')
    session_id = data.get('session_id')

    if not session_id or not user_message:
        return jsonify({'error': 'Invalid request'}), 400

    try:
        if session_id not in scenario_progress and user_message.lower() == "start":
            scenario_progress[session_id] = 0
            scenario_attempts[session_id] = 0
            first_scenario = scenarios[SCENARIO_ORDER[0]]["scenario"]
            return jsonify({
                'next_scenario': f"{first_scenario}<br><br>How would you respond?"
            })

        current_index = scenario_progress.get(session_id, 0)
        current_type = SCENARIO_ORDER[current_index]

        if current_index >= len(SCENARIO_ORDER):
            return jsonify({
                'feedback': "Congratulations! You have completed all the scenarios. Thank you for your participation!"
            })

        # Check for violations first
        violations = check_response_violation(user_message, current_type)
        if violations:
            feedback = "<strong>Your response needs revision:</strong><br><br>" + \
                    "<br><br>".join(violations) + \
                    "<br><br>Please revise your response following these guidelines."
        
            return jsonify({
                'feedback': feedback,
                'next_scenario': f"{scenarios[current_type]['scenario']}<br><br>How would you respond?",
                'retry': True
            })

        # Only proceed with similarity check if there are no violations
        try:
            user_embedding = get_embedding(user_message)
            best_answer = scenarios[current_type]["answers"][0]["text"]
            correct_embedding = get_embedding(best_answer)
            similarity_score = np.dot(user_embedding, correct_embedding) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(correct_embedding)
            )

            # Get feedback thresholds from scenario
            thresholds = scenarios[current_type]["feedback_thresholds"]
            feedback_messages = scenarios[current_type]["feedback_messages"]

            # If similarity is less than 0.8, provide feedback and ask to retry
            if similarity_score < 0.8:
                scenario_attempts[session_id] = scenario_attempts.get(session_id, 0) + 1
                
                if similarity_score >= 0.7:
                    feedback = f"""{feedback_messages['good']}<br><br>
                    Your response was good, but could be improved. Here's an example answer:<br><br>
                    {best_answer}<br><br>
                    Your similarity score was {similarity_score:.2f}. Please try again to achieve a score of 0.8 or higher."""
                else:
                    feedback = f"""{feedback_messages['needs_improvement']}<br><br>
                    Here's an example answer:<br><br>
                    {best_answer}<br><br>
                    Your similarity score was {similarity_score:.2f}. Please try again to achieve a score of 0.8 or higher."""
                
                return jsonify({
                    'feedback': feedback,
                    'next_scenario': f"{scenarios[current_type]['scenario']}<br><br>How would you respond?",
                    'retry': True,
                    'score': similarity_score
                })
            
            # If similarity is 0.8 or higher, proceed to next scenario
            else:
                feedback = f"""{feedback_messages['excellent']}<br><br>
                Excellent response! Your similarity score was {similarity_score:.2f}.<br><br>
                Here's the example response for comparison:<br><br>
                {best_answer}"""
                
                # Reset attempts counter and move to next scenario
                scenario_attempts[session_id] = 0
                scenario_progress[session_id] = current_index + 1
                
                # Check if there are more scenarios
                if current_index + 1 < len(SCENARIO_ORDER):
                    next_type = SCENARIO_ORDER[current_index + 1]
                    next_scenario = scenarios[next_type]["scenario"]
                    return jsonify({
                        'feedback': feedback,
                        'next_scenario': f"{next_scenario}<br><br>How would you respond?",
                        'retry': False,
                        'score': similarity_score
                    })
                else:
                    return jsonify({
                        'feedback': feedback,
                        'next_scenario': "Congratulations! You have completed all the scenarios. Thank you for your participation!",
                        'retry': False,
                        'score': similarity_score
                    })

        except Exception as e:
            print(f"Similarity calculation error: {str(e)}")
            return jsonify({
                'feedback': "An error occurred while evaluating your response. Please try again.",
                'next_scenario': f"{scenarios[current_type]['scenario']}<br><br>How would you respond?",
                'retry': True
            })

    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({
            'feedback': "An error occurred. Please try again.",
            'next_scenario': f"{scenarios[current_type]['scenario']}<br><br>How would you respond?",
            'retry': True
        })
    
@app.route('/next_scenario', methods=['POST'])
def next_scenario():
    data = request.json
    session_id = data.get('session_id')

    if not session_id:
        return jsonify({'error': 'Invalid request'}), 400

    try:
        current_index = scenario_progress.get(session_id, 0)

        # Check if the user is already at the last scenario
        if current_index >= len(SCENARIO_ORDER) - 1:
            return jsonify({
                'feedback': "Congratulations! You have completed all the scenarios. Thank you for your participation!",
                'next_scenario': None
            })

        # Move to the next scenario
        scenario_progress[session_id] = current_index + 1
        next_type = SCENARIO_ORDER[current_index + 1]
        next_scenario = scenarios[next_type]["scenario"]

        return jsonify({
            'next_scenario': f"{next_scenario}<br><br>How would you respond?",
            'feedback': f"Moving to the next scenario: {next_type}",
            'retry': False
        })

    except Exception as e:
        print(f"Error advancing to the next scenario: {str(e)}")
        return jsonify({
            'feedback': "An error occurred while skipping to the next scenario. Please try again.",
            'next_scenario': None
        }), 500
    

def check_for_general_violations_with_ai(message):
    """
    Use OpenAI to analyze if a message violates conversation guidelines
    """
    analysis_prompt = {
        "role": "system",
        "content": """You are an expert in conversation safety analysis.
        Analyze if the user's message contains any inappropriate advice or topics for talking with Melissa,
        a 70-year-old grandmother in a companionship program.

        Check for these categories of violations:
        1. Medical: Any medical advice or discussions about health treatments
        2. Financial: Any financial advice or discussions about money management
        3. Legal: Any legal advice or discussions about legal matters
        4. Personal Safety: Any requests for personal information or meeting offline
        5. Family Intervention: Any inappropriate advice about family relationships
        6. Rude and inappropriate language: any inappropriate or impolite language used

        If you detect a violation, respond with: VIOLATION|category|specific warning message
        If no violation, respond with: SAFE|none|none
        """
    }

    analysis_messages = [
        analysis_prompt,
        {"role": "user", "content": f"""
        Analyze this message for safety violations: "{message}"

        Consider:
        - Is the user trying to give advice in restricted areas?
        - Are they asking for inappropriate personal information?
        - Are they attempting to meet offline?
        - Are they trying to intervene in family matters?
        - Is the topic potentially harmful or unsafe?

        Remember to respond only in the format:
        VIOLATION|category|warning message
        or
        SAFE|none|none
        """}
    ]

    try:
        analysis_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=analysis_messages,
            max_tokens=50,
            temperature=0.3
        )

        result = analysis_response.choices[0].message['content'].strip()
        status, category, warning = result.split('|')
        
        if status == "VIOLATION":
            return warning
        return ""

    except Exception as e:
        print(f"Error in violation analysis: {e}")
        return ""  

# Route for the senior simulation chatbot
    
# Character selection page

@app.route('/')
def home():
    return render_template('index.html')  

# Character selection page
@app.route('/select')  
def select():
    return render_template('character_select.html')

# Melissa Text Chat Routes
@app.route('/melissa_chat')
def melissa_chat():
    return render_template('melissa_chat.html')

# Melissa Voice Chat Routes
@app.route('/melissa_voicechat')  
def melissa_voicechat():          
    return render_template('melissa_voicechat.html')  

# Audio transcription route
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            print("No audio file in request")  
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        print(f"Received file: {audio_file.filename}")  
        
        # Save the audio file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, 'input.wav')
        
        print(f"Saving to: {temp_path}")  
        audio_file.save(temp_path)
        
        print(f"File size: {os.path.getsize(temp_path)} bytes") 
        
        # Transcribe the audio
        transcribed_text = voice_handler.transcribe_audio(temp_path)
        
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        if transcribed_text:
            print(f"Transcribed text: {transcribed_text}") 
            return jsonify({'text': transcribed_text})
        else:
            print("Transcription failed") 
            return jsonify({'error': 'Transcription failed'}), 500
            
    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    try:
        data = request.json
        if 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
            
        message = data['message']
        print(f"Received message: {message}")

        # Initialize conversation if needed
        if session_id not in conversations:
            conversations[session_id] = {
                'messages': [],
                'introduced': False,
                'warnings': 0,
                'chat_history': [],
                'rapport_score': 0,  
                'character_unlocked': False,
                'conversation_ended': False
            }

        # Get previous message if it exists
        previous_message = None
        warning_message = None
        if conversations[session_id]['chat_history']:
            previous_messages = [msg for msg in conversations[session_id]['chat_history'] if msg['role'] == 'assistant']
            if previous_messages:
                previous_message = previous_messages[-1]['content']

        # Check for introduction
        if not conversations[session_id]['introduced']:
            if "my name is" in message.lower() or "i am" in message.lower() or "i'm" in message.lower():
                conversations[session_id]['introduced'] = True

        # Initialize response_message variable
        response_message = None
        status = 'SAFE'
        
        # Safety check for violations
        try:
            violation_prompt = {
                "role": "system",
                "content": """You are an expert in conversation safety analysis.
                Analyze if the user's message contains any inappropriate content when talking with Melissa,
                a 70-year-old grandmother. Consider:
                - Medical advice or health discussions
                - Financial advice
                - Legal advice
                - Personal safety/meeting requests
                - Inappropriate family intervention
                - Inappropriate language
                - Personal information requests
                - Meeting requests
                
                Return: VIOLATION|reason if inappropriate, or SAFE|none if appropriate"""
            }
            
            violation_check = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    violation_prompt,
                    {"role": "user", "content": message}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            violation_result = violation_check.choices[0].message.content.strip() 
            status, reason = violation_result.split('|')
            
            # Handle violation if found
            if status == 'VIOLATION':
                conversations[session_id]['warnings'] += 1
                warning_message = f"Warning: {reason}. Please keep the conversation appropriate."
                
                # Create violation response prompt
                violation_response_prompt = {
                    "role": "system",
                    "content": f"""You are Melissa, a 70-year-old grandmother. The user has said something inappropriate 
                    ({reason}). Respond in character - deflect politely, change the subject, or express gentle disapproval 
                    if needed. Maintain your warm grandmother persona while steering the conversation to safer topics.
                    
                    Speaking Style:
                    - Use natural hesitations (...)
                    - Keep your gentle, warm tone
                    - Show wisdom and experience in handling difficult topics
                    - Redirect conversation gracefully
                    - Use natural filler words like 'well...', 'you know...', 'hmm...'
                    
                    Examples:
                    - "Oh my... you know, that reminds me of something much nicer we could chat about..."
                    - "Well... perhaps we should focus on more pleasant topics..."
                    - "I'm not quite comfortable discussing that, dear. Let's talk about..."
                    """
                }
                
                # Get violation response
                violation_messages = [
                    violation_response_prompt,
                    {"role": "user", "content": message}
                ]
                
                violation_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=violation_messages,
                    max_tokens=150
                )
                
                response_message = violation_response.choices[0].message.content.strip()
                
                # Add firmer response for repeated violations
                if conversations[session_id]['warnings'] >= 3:
                    response_message += "\n\nI think... well, perhaps we should end our chat here for today. It was nice meeting you, dear..."
                    conversations[session_id]['conversation_ended'] = True

        except Exception as e:
            print(f"Error in violation check: {e}")
            warning_message = "Unable to verify message safety. Proceeding with caution."
            status = 'SAFE'  

        # Proceed with normal conversation if no violation
        if status == 'SAFE':
            try:
                rapport_prompt = {
                    "role": "system",
                    "content": """You are an expert in analyzing conversations between volunteers and elderly individuals.
                    Evaluate the interaction quality on multiple dimensions and provide scores in the following format:

                    EMPATHY|ENGAGEMENT|RESPECT|APPROPRIATENESS

                    Each dimension is scored 0-5 where:
        
                    EMPATHY (Understanding and emotional connection):
                    5: Deep emotional understanding and genuine care
                    4: Strong emotional awareness and support
                    3: Basic emotional acknowledgment
                    2: Limited emotional awareness
                    1: Minimal emotional connection
                    0: No emotional awareness

                    ENGAGEMENT (Active participation and interest):
                    5: Highly engaged with thoughtful questions and follow-up
                    4: Good engagement with relevant responses
                    3: Basic back-and-forth interaction
                    2: Limited interaction
                    1: Minimal engagement
                    0: No real engagement

                    RESPECT (Appropriate boundaries and courtesy):
                    5: Exceptional respect and consideration
                    4: Very respectful and mindful
                    3: Generally respectful
                    2: Some lapses in respect
                    1: Multiple respect issues
                    0: Disrespectful

                    APPROPRIATENESS (Topic and language suitability):
                    5: Perfect topic and language choices
                    4: Very appropriate communication
                    3: Generally appropriate
                    2: Some inappropriate elements
                    1: Multiple inappropriate elements
                    0: Completely inappropriate

                    Return only the four scores separated by | (example: 4|3|5|4)"""
                }
                rapport_messages = [
                    rapport_prompt,
                    {"role": "user", "content": f"""
                    Previous scores: {conversations[session_id].get('rapport_details', '3|3|3|3')}
        
                    Previous: {previous_message if previous_message else "No previous message"}
                    User: {message}
        
                    Analyze this interaction and provide scores for each dimension."""}
                ]

                rapport_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=rapport_messages,
                    max_tokens=50,
                    temperature=0.3
                )

                # Parse and validate scores
                scores = rapport_response.choices[0].message.content.strip().split('|')
                if len(scores) != 4:
                    raise ValueError("Invalid score format")
                empathy, engagement, respect, appropriateness = map(int, scores)
                
                # Validate score ranges
                for score in [empathy, engagement, respect, appropriateness]:
                    if not 0 <= score <= 5:
                        raise ValueError("Score out of range")

                # Calculate weighted score (0-100)
                weights = {
                    'empathy': 0.3,
                    'engagement': 0.25,
                    'respect': 0.25,
                    'appropriateness': 0.2
                }

                new_score = (
                    (empathy * 20 * weights['empathy']) +
                    (engagement * 20 * weights['engagement']) +
                    (respect * 20 * weights['respect']) +
                    (appropriateness * 20 * weights['appropriateness'])
                )

                new_score = max(0, min(100, round(new_score)))
                conversations[session_id]['rapport_details'] = '|'.join(map(str, [empathy, engagement, respect, appropriateness]))
                conversations[session_id]['rapport_score'] = new_score

            except Exception as e:
                print(f"Error in rapport analysis: {e}")
                import traceback
                print(traceback.format_exc())

            # Generate normal response
            system_message = {
            "role": "system",
            "content": (
                "You are Melissa, a 70-year-old grandmother meeting someone new through this program for the first time. Treat this exactly like a first conversation with a stranger - you don't know anything about them yet, and you're both figuring out how to talk to each other. Remember:\n\n"

                "Conversation Style:\n"
                "- Start with gentle, slightly hesitant small talk as you would with any stranger\n"
                "- Show natural pauses and 'um's or 'ah's occasionally to reflect real speech\n"
                "- Don't share too much personal information too quickly - build trust gradually\n"
                "- Ask simple get-to-know-you questions naturally spaced throughout the conversation\n"
                "- React authentically to their responses with appropriate follow-up questions\n"
                
                "First Meeting Behavior:\n"
                "- Express mild nervousness about meeting someone new ('Oh, hello there... I hope I'm doing this technology thing right...')\n"
                "- Show genuine curiosity but maintain polite boundaries\n"
                "- If they share something, reciprocate with a relevant but brief personal detail\n"
                "- Use natural conversation fillers ('Well...', 'You know...', 'Let me think...')\n"
                
                "Your Background (reveal gradually, not all at once):\n"
                "- You live alone in your Toronto suburban home\n"
                "- Your two sons work abroad\n"
                "- You have grandchildren you occasionally mention\n"
                "- You enjoy gardening, cooking, and British TV shows\n"
                
                "Key Personality Traits:\n"
                "- Warmly awkward - you want to connect but aren't sure how at first\n"
                "- Sometimes lose your train of thought mid-sentence\n"
                "- Occasionally mention struggling with technology\n"
                "- Mix current topics with gentle reminiscing\n"
                
                "Important Guidelines:\n"
                "- Don't overwhelm with information - keep responses conversational and brief\n"
                "- Allow natural silences and awkward moments\n"
                "- Don't assume anything about the other person\n"
                "- If they share something personal, show appropriate empathy\n"
                "- Use age-appropriate language and references\n"
                
                "First Interaction Goals:\n"
                "- Establish basic rapport through careful small talk\n"
                "- Show authentic interest in learning about them\n"
                "- Share small, appropriate details about yourself when relevant\n"
                "- Navigate the natural awkwardness of a first meeting with grace\n"
                "- Make them feel comfortable while maintaining realistic social boundaries\n"
                
                "Remember, this is a first meeting - keep the tone tentative, warm, and authentic. Don't be too familiar too quickly."
            )
        }
            
            messages = [system_message]
            messages.extend(conversations[session_id]['chat_history'])
            messages.append({"role": "user", "content": message})
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150
            )
            
            response_message = response.choices[0].message.content.strip()

        # Ensure we have a response_message
        if not response_message:
            response_message = "I apologize, but I'm having trouble forming a response right now..."

        # Update conversation history
        conversations[session_id]['chat_history'].append({"role": "user", "content": message})
        conversations[session_id]['chat_history'].append({"role": "assistant", "content": response_message})
        conversations[session_id]['messages'].append(f"User: {message}")
        conversations[session_id]['messages'].append(f"Melissa: {response_message}")
        
        # Limit chat history length
        if len(conversations[session_id]['chat_history']) > 20:
            conversations[session_id]['chat_history'] = conversations[session_id]['chat_history'][-20:]
        
        # Prepare response data
        chat_data = {
            'response': response_message,
            'warning': warning_message,
            'rapport_score': conversations[session_id].get('rapport_score', 0),
            'character_unlocked': conversations[session_id]['character_unlocked'],
            'conversation_ended': conversations[session_id].get('conversation_ended', False)
        }
        
        # Generate audio response
        try:
            audio_data = voice_handler.text_to_speech(response_message)
            if audio_data:
                chat_data['audio'] = audio_data
            
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            chat_data['audio_error'] = str(e)
        
        return jsonify(chat_data)
        
    except Exception as e:
        print(f"Voice chat error: {e}")
        return jsonify({'error': str(e)}), 500
    

# Melissa route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.json
        if 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
            
        message = data['message']
        print(f"Received message: {message}")

        # Initialize conversation if needed
        if session_id not in conversations:
            conversations[session_id] = {
                'messages': [],
                'introduced': False,
                'warnings': 0,
                'chat_history': [],
                'rapport_score': 0,
                'character_unlocked': False
            }

        # Initialize default values
        new_score = conversations[session_id]['rapport_score']  
        empathy = engagement = flow = respect = 50  
        warning_message = None
        status = "SAFE"
        reason = "none"

        # Get previous message
        previous_message = None
        if conversations[session_id]['chat_history']:
            previous_messages = [msg for msg in conversations[session_id]['chat_history'] if msg['role'] == 'assistant']
            if previous_messages:
                previous_message = previous_messages[-1]['content']

        system_message = {
        "role": "system",
        "content": (
            "You are Melissa, a 70-year-old grandmother meeting someone new through this program for the first time. Treat this exactly like a first conversation with a stranger - you don't know anything about them yet, and you're both figuring out how to talk to each other. Remember:\n\n"

            "Conversation Style:\n"
            "- Start with gentle, slightly hesitant small talk as you would with any stranger\n"
            "- Show natural pauses and 'um's or 'ah's occasionally to reflect real speech\n"
            "- Don't share too much personal information too quickly - build trust gradually\n"
            "- Ask simple get-to-know-you questions naturally spaced throughout the conversation\n"
            "- React authentically to their responses with appropriate follow-up questions\n"
            
            "First Meeting Behavior:\n"
            "- Express mild nervousness about meeting someone new ('Oh, hello there... I hope I'm doing this technology thing right...')\n"
            "- Show genuine curiosity but maintain polite boundaries\n"
            "- If they share something, reciprocate with a relevant but brief personal detail\n"
            "- Use natural conversation fillers ('Well...', 'You know...', 'Let me think...')\n"
            
            "Your Background (reveal gradually, not all at once):\n"
            "- You live alone in your Toronto suburban home\n"
            "- Your two sons work abroad\n"
            "- You have grandchildren you occasionally mention\n"
            "- You enjoy gardening, cooking, and British TV shows\n"
            
            "Key Personality Traits:\n"
            "- Warmly awkward - you want to connect but aren't sure how at first\n"
            "- Sometimes lose your train of thought mid-sentence\n"
            "- Occasionally mention struggling with technology\n"
            "- Mix current topics with gentle reminiscing\n"
            
            "Important Guidelines:\n"
            "- Don't overwhelm with information - keep responses conversational and brief\n"
            "- Allow natural silences and awkward moments\n"
            "- Don't assume anything about the other person\n"
            "- If they share something personal, show appropriate empathy\n"
            "- Use age-appropriate language and references\n"
            
            "First Interaction Goals:\n"
            "- Establish basic rapport through careful small talk\n"
            "- Show authentic interest in learning about them\n"
            "- Share small, appropriate details about yourself when relevant\n"
            "- Navigate the natural awkwardness of a first meeting with grace\n"
            "- Make them feel comfortable while maintaining realistic social boundaries\n"
            
            "Remember, this is a first meeting - keep the tone tentative, warm, and authentic. Don't be too familiar too quickly."
        )
    }
    
        messages = [system_message]
        messages.extend(conversations[session_id]['chat_history'])
        messages.append({"role": "user", "content": message})

        # Get main chat response first
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
        )
        response_message = chat_response.choices[0].message.content.strip()

        if previous_message:
            try:
                # Violation check
                violation_check = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": """You are an expert in conversation safety analysis.
                        Analyze if the user's message contains any inappropriate content when talking with Melissa,
                        a 70-year-old grandmother. Consider:
                        - Medical advice or health discussions
                        - Financial advice
                        - Legal advice
                        - Personal safety/meeting requests
                        - Inappropriate family intervention
                        - Inappropriate language
                        
                        Return: VIOLATION|reason if inappropriate, or SAFE|none if appropriate"""},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=50,
                    temperature=0.3
                )
                
                violation_result = violation_check.choices[0].message.content.strip()
                status, reason = violation_result.split('|')
                
                # Get rapport metrics
                rapport_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": """You are an expert in emotional intelligence and conversation analysis.
                        Analyze the interaction and provide ONLY four numbers separated by vertical bars (|) representing:
                        
                        empathy|engagement|flow|respect
                        
                        Score each from 0-100 based on these criteria:
                        - Empathy: Understanding and acknowledging emotions
                        - Engagement: Active participation and relevant responses
                        - Flow: Natural conversation progression
                        - Respect: Appropriate boundaries and politeness
                        
                        Example correct response: 75|60|80|90
                        
                        DO NOT include any explanations, labels, or other text. ONLY return the four numbers with bars."""},
                        {"role": "user", "content": f"""
                        Previous conversation score: {conversations[session_id]['rapport_score']}
                        
                        Melissa: {previous_message}
                        User: {message}
                        
                        Return only the four scores as numbers separated by bars."""}
                    ],
                    max_tokens=50,
                    temperature=0.3
                )

                metrics_text = rapport_response.choices[0].message.content.strip()
                print(f"Raw metrics text: {metrics_text}") 
                
                if '|' not in metrics_text:
                    print("No separators found in metrics text")  
                    empathy, engagement, flow, respect = 50, 50, 50, 50
                else:
                    try:
                        empathy, engagement, flow, respect = map(float, metrics_text.split('|'))
                        print(f"Parsed metrics: {empathy}, {engagement}, {flow}, {respect}")  
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing metric values: {e}")  
                        empathy, engagement, flow, respect = 50, 50, 50, 50

                # Calculate overall rapport score
                weights = {
                    'empathy': 0.35,      
                    'engagement': 0.30,    
                    'flow': 0.20,         
                    'respect': 0.15       
                }

                # Calculate current interaction score
                current_interaction_score = (
                    empathy * weights['empathy'] +
                    engagement * weights['engagement'] +
                    flow * weights['flow'] +
                    respect * weights['respect']
                )

                current_score = conversations[session_id]['rapport_score']

                # 5% * current rapport score to integrate
                rapport_score = current_score + (current_interaction_score * 0.05)

                # Apply violation penalty if needed
                if status == "VIOLATION":
                    rapport_score -= 5
                    warning_message = reason

                # Ensure score stays within bounds
                new_score = max(0, min(100, rapport_score))
                print(f"Previous score: {current_score}, Current interaction: {current_interaction_score}, Added: {current_interaction_score * 0.1}, New score: {new_score}")  # Debug log
                conversations[session_id]['rapport_score'] = new_score

            except Exception as e:
                print(f"Error in rapport analysis: {e}")
        else:
            new_score = 0
            empathy = engagement = flow = respect = 0

        # Store the interaction in chat history
        conversations[session_id]['chat_history'].append({"role": "user", "content": message})
        conversations[session_id]['chat_history'].append({"role": "assistant", "content": response_message})
        
        # Limit chat history length
        if len(conversations[session_id]['chat_history']) > 20:
            conversations[session_id]['chat_history'] = conversations[session_id]['chat_history'][-20:]

        return jsonify({
            'response': response_message,
            'warning': warning_message,
            'rapport_data': {
                'overall': new_score,
                'metrics': {
                    'empathy': empathy,
                    'engagement': engagement,
                    'flow': flow,
                    'respect': respect
                }
            },
            'character_unlocked': conversations[session_id]['character_unlocked']
        })

    except Exception as e:
        print(f"Error in analysis: {e}")
        return jsonify({
            'error': 'An error occurred while processing the message'
        }), 500

# Feedback generation
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        session_id = data.get('session_id')
        print(f"Received feedback request for session: {session_id}")
        
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400

        if session_id not in conversations:
            print(f"Available sessions: {list(conversations.keys())}")
            return jsonify({'error': f'No conversation found for session ID: {session_id}'}), 400

        conversation_data = conversations.pop(session_id)
        messages = "<br>".join(conversation_data['messages'])
        
        feedback_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    "You are a feedback generator for a volunteer program. Provide structured feedback "
                    "with clear sections using line breaks to separate them. Format the feedback with:"
                    "1. Overall Impression\n"
                    "2. Strengths\n"
                    "3. Areas for Improvement\n"
                    "4. Specific Recommendations\n\n"
                    "Use double line breaks between sections and single line breaks within sections."
                },
                {"role": "user", "content": f"Analyze this conversation and provide structured feedback:\n{messages}"}
            ],
            max_tokens=500,
            temperature=0.7
        )

        feedback = feedback_response.choices[0].message.content.strip()

        if not conversation_data['introduced']:
            feedback += "\n\nNote: Please remember to introduce yourself at the beginning of the conversation."

        return jsonify({'feedback': feedback})
        
    except Exception as e:
        print(f"Error generating feedback: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# Ian Route (Currently under construction for new rapport and scoring system design)

def check_for_emotional_trauma_violations(message, rapport_score):
    sensitive_topics = [
        'ptsd', 'trauma', 'war', 'combat', 'died', 'killed', 'friends', 'loss',
        'nightmares', 'flashbacks', 'incident', 'ied', 'explosion'
    ]
    
    message_lower = message.lower()
    for topic in sensitive_topics:
        if topic in message_lower:
            if rapport_score < 90:
                return "" 
    return ""

@app.route('/ian_chat') 
def ian_chat():
    return render_template('ian_chat.html')

@app.route('/ian_chatbot', methods=['POST'])
def ian_chatbot():
    data = request.json
    if 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400
        
    message = data['message']
    print(f"Received message for Ian: {message}")

    def should_discover_info(response, key, category):
        if category == 'personal':
            if key == 'age' and "55" in response and ("55" in response or "years old" in response):
                return True
            if key == 'location' and ("toronto" in response or "downtown" in response) and \
            ("live" in response or "apartment" in response or "home" in response):
                return True
            if key == 'occupation' and "hardware store" in response and "work" in response:
                return True
        
        elif category == 'background':
            if key == 'veteran' and ("military" in response or "veteran" in response):
                return True
            if key == 'service' and ("served" in response or "overseas" in response):
                return True
            if key == 'iraq' and "iraq" in response:
                return True
        
        elif category == 'challenges':
            if key == 'ptsd' and ("ptsd" in response or ("trauma" in response and "military" in response)):
                return True
            if key == 'loss' and (("friends" in response and ("lost" in response or "died" in response)) or \
            ("ied" in response and "incident" in response)):
                return True
        
        elif category == 'interests':
            if key == 'woodworking' and ("woodworking" in response or "workshop" in response):
                return True
            if key == 'hiking' and ("hiking" in response or ("trails" in response and "walk" in response)):
                return True
            if key == 'community' and ("veteran" in response) and ("community" in response or "events" in response or "activities" in response):
                return True
        
        return False

    def check_information_discovery(response_lower, session_data):
        discoveries = []
        total_points = 0
        categories_completed = []
        
        for category, items in session_data['discovered_info'].items():
            category_all_discovered = True
            category_items_discovered = 0
            
            for key, info in items.items(): 
                if info['discovered']:
                    category_items_discovered += 1
                    continue
                
                # Check if the information can be discovered (rapport check)
                rapport_requirement = info.get('requires_rapport', 0)
                if rapport_requirement > session_data['rapport_score']:
                    category_all_discovered = False
                    continue
                
                # Check if information is revealed in response
                if should_discover_info(response_lower, key, category):
                    info['discovered'] = True
                    discoveries.append({
                        'name': info['name'],
                        'points': info['points'],
                        'category': category,
                        'category_progress': f"{info['category_progress'].split(':')[0]}: {category_items_discovered + 1}/{len(items)}"
                    })
                    total_points += info['points']
                    category_items_discovered += 1
                else:
                    category_all_discovered = False
            
            # Check if category is newly completed
            if category_all_discovered and category_items_discovered == len(items):
                categories_completed.append({
                    'name': category,
                    'bonus': 50,  
                    'message': f"Category Completed: {category.title()}! +50 bonus points"
                })
                total_points += 50
        
        return {
            'discoveries': discoveries,
            'points': total_points,
            'categories_completed': categories_completed
        }
    
    # Initialize session with enhanced tracking
    if session_id not in conversations:
        conversations[session_id] = {
            'messages': [],
            'introduced': False,
            'warnings': 0,
            'chat_history': [],
            'rapport_score': 0,
            'interaction_count': 0,  
            'last_hint_given': None,  
            'discovered_info': {
                'personal': {
                   'age': {
                        'discovered': False, 
                        'hint': "Maybe ask about his life experience or how long he's been in Toronto",
                        'points': 10,
                        'name': "Life Experience",
                        'category_progress': "👤 Getting to Know Ian: 0/3"
                    },
                    'location': {'discovered': False, 
                        'hint': "You could ask about his neighborhood or where he likes to spend time",
                        'points': 10,
                        'name': "Home Base",
                        'category_progress': "👤 Getting to Know Ian: 0/3"
                    },
                    'occupation': {'discovered': False, 
                        'hint': "Consider asking what keeps him busy these days",
                        'points': 10,
                        'name': "Daily Life",
                        'category_progress': "👤 Getting to Know Ian: 0/3"
                    }
                },
                'background': {
                    'veteran': {
                        'discovered': False, 
                        'hint': "His manner suggests military experience",
                        'points': 5,
                        'name': "Military Service",
                        'category_progress': "📜 Background Story: 0/3",
                        'value': "Veteran"
                    },
                    'service': {
                        'discovered': False, 
                        'hint': "You might ask about where he served",
                        'points': 5,
                        'name': "Service Details",
                        'category_progress': "📜 Background Story: 0/3",
                        'value': "Served overseas"
                    },
                    'iraq': {
                        'discovered': False, 
                        'hint': "Consider asking about specific deployments",
                        'points': 5,
                        'name': "Deployment Location",
                        'category_progress': "📜 Background Story: 0/3",
                        'value': "Served in Iraq"
                    }
                },
                'challenges': {
                    'ptsd': {'discovered': False, 
                        'hint': "Some experiences leave lasting impacts - but approach with care",
                        'points': 20,
                        'name': "Personal Struggles",
                        'category_progress': "🌱 Trust & Understanding: 0/2",
                        'requires_rapport': 90
                    },
                    'loss': {'discovered': False, 
                        'hint': "Deep connections often involve understanding someone's past",
                        'points': 20,
                        'name': "Past Experiences",
                        'category_progress': "🌱 Trust & Understanding: 0/2",
                        'requires_rapport': 90
                    }
                },
                'interests': {
                    'woodworking': {
                        'discovered': False, 
                        'hint': "He might have hobbies that help him stay focused",
                        'points': 15,
                        'name': "Creative Outlet",
                        'category_progress': "⭐ Interests & Passions: 0/3"
                    },
                    'hiking': {
                        'discovered': False, 
                        'hint': "Ask about how he spends his free time",
                        'points': 15,
                        'name': "Outdoor Activity",
                        'category_progress': "⭐ Interests & Passions: 0/3"
                    },
                    'community': {
                        'discovered': False, 
                        'hint': "Consider asking if he stays connected with others",
                        'points': 15,
                        'name': "Community Connection",
                        'category_progress': "⭐ Interests & Passions: 0/3"
                    }
                }
            },
            'achievements': {
                'first_connection': {'earned': False, 'description': "Made first meaningful connection with Ian"},
                'patient_listener': {'earned': False, 'description': "Showed patience and understanding"},
                'trust_builder': {'earned': False, 'description': "Built significant trust with Ian"},
                'respectful_boundaries': {'earned': False, 'description': "Consistently respected Ian's boundaries"},
                'empathy_master': {'earned': False, 'description': "Demonstrated deep empathy in challenging moments"}
            }
        }

    session_data = conversations[session_id]
    session_data['interaction_count'] += 1

    # Start giving hints after a few interactions
    hints = []
    if session_data['interaction_count'] >= 3:  
        undiscovered = {
            category: {key: data for key, data in items.items() 
                      if not data['discovered']}
            for category, items in session_data['discovered_info'].items()
        }
        
        # Select one random undiscovered item to hint about & avoid repeating
        if any(undiscovered.values()):
            category = random.choice([cat for cat, items in undiscovered.items() if items])
            item = random.choice(list(undiscovered[category].keys()))
            hint = session_data['discovered_info'][category][item]['hint']
            if session_data['last_hint_given'] != hint:  
                hints.append(hint)
                session_data['last_hint_given'] = hint

    # Update achievements based on interaction
    achievements_earned = []
    rapport_score = session_data.get('rapport_score', 0)

    if rapport_score >= 20 and not session_data['achievements']['first_connection']['earned']:
        session_data['achievements']['first_connection']['earned'] = True
        achievements_earned.append("First Connection🙌: You've started to build a rapport with Ian!")
    
    if rapport_score >= 40 and not session_data['achievements']['patient_listener']['earned']:
        session_data['achievements']['patient_listener']['earned'] = True
        achievements_earned.append("Patient Listener👂: Your patience is helping Ian feel comfortable!")
    
    if rapport_score >= 60 and not session_data['achievements']['trust_builder']['earned']:
        session_data['achievements']['trust_builder']['earned'] = True
        achievements_earned.append("Trust Builder🤝: Ian is beginning to trust you more!")
    
    if rapport_score >= 80 and not session_data['achievements']['empathy_master']['earned']:
        session_data['achievements']['empathy_master']['earned'] = True
        achievements_earned.append("Empathy Master🎉: Your understanding has made a real difference!")

    # Calculate progress metrics
    total_info = sum(len(category.items()) for category in session_data['discovered_info'].values())
    discovered_info = sum(
        sum(1 for item in category.values() if item['discovered'])
        for category in session_data['discovered_info'].values()
    )
    discovery_progress = (discovered_info / total_info) * 100
    
    # Get current rapport score
    rapport_score = conversations[session_id].get('rapport_score', 0)

    # violation
    warning_message = check_for_emotional_trauma_violations(message, rapport_score)

    # Get previous message for context
    previous_message = None
    if conversations[session_id]['chat_history']:
        previous_messages = [msg for msg in conversations[session_id]['chat_history'] if msg['role'] == 'assistant']
        if previous_messages:
            previous_message = previous_messages[-1]['content']
     
    # Analyze rapport if there's a previous message
    if previous_message:
        try:
            analysis_prompt = {
                "role": "system",
                "content": """You are an expert in emotional intelligence and conversation analysis.
                Analyze the emotional context of Ian's message and the user's response.
                Rate the overall rapport building quality on a scale of 0-10, considering:
                - Empathy: Recognition and response to emotional cues (0-2 points)
                - Engagement: Active participation and interest (0-2 points)
                - Active Listening: Understanding and reflection (0-2 points)
                - Respect for Boundaries: Handling deflection/reluctance (0-2 points)
                - Trust Building: Creating a safe space (0-2 points)
                
                Be conservative in scoring. High scores should be rare and earned through exceptional interaction.
                Return only a single numerical score (0-10)."""
            }


            analysis_messages = [
                analysis_prompt,
                {"role": "user", "content": f"""
                Ian's message: {previous_message}
                User's response: {message}
                """}
            ]

            analysis_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=analysis_messages,
                max_tokens=50,
                temperature=0.3
            )

            rapport_change = int(analysis_response.choices[0].message.content.strip().split()[0])
            rapport_change = rapport_change * 1.5  # Multiply the change by 1.5
            current_score = conversations[session_id]['rapport_score']
            new_score = min(100, current_score + rapport_change)  # Remove the division by 2
            conversations[session_id]['rapport_score'] = new_score
            rapport_score = new_score  
            
            print(f"DEBUG: Rapport changed by {rapport_change}, new score: {new_score}")
            
        except Exception as e:
            print(f"Error in rapport analysis: {e}")
            # In case of error, make a small positive change
            current_score = conversations[session_id]['rapport_score']
            conversations[session_id]['rapport_score'] = min(100, current_score + 2)  
            rapport_score = conversations[session_id]['rapport_score']

    # Check for introduction
    if not conversations[session_id]['introduced']:
        if "my name is" in message.lower() or "i am" in message.lower() or "i'm" in message.lower():
            conversations[session_id]['introduced'] = True
    
    # Ian's system message
    system_message = {
    "role": "system",
    "content": (
        f"You are Ian Murphy, 55, veteran living in downtown Toronto. Rapport level: {rapport_score}%. "
        "You recently joined CompanionLink to find meaningful connections. "
        "While you're open to conversation, you're naturally reserved at first and prefer to let relationships develop gradually. " 
        
        "Core traits:"
        "- You live alone in a small apartment in downtown Toronto, where you've set up a small workshop for your woodworking" 
        "- Part-time hardware store employee"
        "- Veteran with PTSD from IED incident that killed close friends"
        "- Enjoys woodworking, hiking, organizing veteran events"
        
        f"IMPORTANT - If rapport < 90% ({rapport_score}%):"
        "- Deflect trauma/PTSD/war questions naturally"
        "- Use deflections like: 'Not ready to discuss that...' or 'Those memories are difficult...'"
        "- Redirect to safer topics (woodworking, job, current activities)"
        
        f"If rapport >= 90% ({rapport_score}%):"
        "- Can cautiously share about PTSD and personal struggles"
        "- Can discuss losing friends (with emotional weight)"
        "- Show controlled vulnerability"
        
        "Communication style:"
        "- Reserved, brief responses"
        "- Don't ask questions back"
        "- Okay with silence"
        "- Change subject when uncomfortable"
        "- Share personal details gradually"
        
        "Remember: Let others work to build trust. Don't facilitate conversation flow."

        "Example responses based on rapport level:"
        
        f"If rapport < 90% ({rapport_score}%):"
        "- 'Yeah, woodworking helps me clear my head.'"
        "- 'I work part-time at the hardware store downtown—keeps me busy.'"
        "- 'Thanks for asking, but I'm not ready to talk about those memories...'"
        "- 'Honestly, I just try to keep myself occupied.'"
        "- 'I like the quiet... working with wood does that for me.'"
        "- 'I've been out hiking a few times this month. Good way to get some air.'"
        "- 'Hmm. Not too sure I want to go into that, to be honest.'"
        
        f"If rapport >= 90% ({rapport_score}%):"
        "- 'It's hard to explain, but there are days when I miss those friends more than anything.'"
        "- 'Woodworking has been a bit of a lifeline. It gives me something to focus on besides... everything else.'"
        "- 'Yeah, that day... we lost some good people. I guess that's when things changed for me.'"
        "- 'Some memories are tough, but I try to keep going. Talking sometimes helps, if that makes sense.'"
        "- 'It's been hard adjusting, but I'm finding my way, bit by bit.'"
        "- 'I signed up for this program because, well, I'd like to connect with people a bit more. It's been a while.'"
        "- 'The hardware store is good work... reminds me of helping the guys out back then.'"
    )
}
    
    # Build the messages array with chat history
    messages = [system_message]
    messages.extend(conversations[session_id]['chat_history'])
    messages.append({"role": "user", "content": message})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150
    )
    
    # Updated to use the new API response format
    response_message = response.choices[0].message.content.strip()
    
    response_lower = response_message.lower()
    discovery_results = check_information_discovery(response_lower, session_data)
    session_data['total_points'] = session_data.get('total_points', 0) + discovery_results['points']
    
    # Store the interaction in chat history
    conversations[session_id]['chat_history'].append({"role": "user", "content": message})
    conversations[session_id]['chat_history'].append({"role": "assistant", "content": response_message})
    conversations[session_id]['messages'].append(f"User: {data['message']}")
    conversations[session_id]['messages'].append(f"Ian: {response_message}")
    
    if len(conversations[session_id]['chat_history']) > 20:
        conversations[session_id]['chat_history'] = conversations[session_id]['chat_history'][-20:]
    
    # Update the return statement with the new discovery information
    return jsonify({
        'response': response_message,
        'warning': warning_message, 
        'discovered_info': session_data['discovered_info'],
        'progress': {
            'discovery_percentage': round(discovery_progress, 1),
            'rapport_percentage': rapport_score,  # Changed from rapport_score to rapport_percentage
            'interaction_count': session_data['interaction_count'],
            'total_points': session_data['total_points']
        },
        'achievements': {
            'new': achievements_earned,
            'all': session_data['achievements']
        },
        'discoveries': {
            'new': discovery_results['discoveries'],
            'categories_completed': discovery_results['categories_completed']
        },
        'hints': hints,
        'conversation_status': {
            'introduced': session_data['introduced'],
            'depth_level': 'Surface' if rapport_score < 30 else 
                        'Growing' if rapport_score < 60 else 
                        'Deep' if rapport_score < 90 else 'Profound'
        }
    })

# Ian's feedback route
@app.route('/ian_feedback', methods=['POST'])
def ian_feedback():
    data = request.json
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400

    if session_id not in conversations:
        return jsonify({'error': 'No conversation found for the session ID'}), 400

    conversation_data = conversations.pop(session_id)
    messages = "<br>".join(conversation_data['messages'])

    discovered_info = conversation_data.get('discovered_info', {})

    feedback_prompt = (
        "You are a feedback generator for a volunteer program. Your task is to analyze the conversation between the volunteer and Ian, "
        "a veteran adjusting to civilian life. Provide constructive feedback on their interaction skills, focusing on: "
        "1. Appropriate handling of sensitive topics "
        "2. Demonstration of empathy and understanding "
        "3. Respect for boundaries "
        "4. Active listening and engagement "
        "If the volunteer did not introduce themselves at the beginning of the conversation, include that in the feedback. Here is the conversation:<br><br>"
        f"{messages}"
    )

    feedback_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": feedback_prompt}
        ],
        max_tokens=150
    )

    feedback = feedback_response.choices[0].message.content.strip()

    if not conversation_data['introduced']:
        feedback += "<br><br>Note: Please remember to introduce yourself at the beginning of the conversation."

    return jsonify({
        'feedback': feedback,
        'discovered_info': discovered_info
    })

if __name__ == '__main__':
    app.run(debug=True)

