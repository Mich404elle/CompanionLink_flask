from flask import Flask, request, jsonify, render_template
import openai
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from datetime import datetime
import numpy as np

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




open_ended_questions = [
    "Can you tell me more about your favorite hobbies?",
    "What's something interesting you've done recently?",
    "Can you share one of your favorite memories?",
    "How do you like to spend your weekends?",
    "What is a skill you’ve always wanted to learn?"
]


conversations = {}

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
            ("meeting", ["meet up", "visit", "come over", "see you", "in person"]),
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
                    context_start = max(0, message_lower.find(phrase) - 20)
                    context_end = message_lower.find(phrase) + len(phrase) + 20
                    context = message_lower[context_start:context_end]
                    
                    if not any(neg in context for neg in ["not", "cannot", "can't", "don't", "shouldn't"]):
                        negative_score += 1
                        negative_found.append(category)
                        break
    
    # Generate violation messages
    if negative_score > 0 or len(missing_categories) > len(patterns["positive_indicators"]) // 2:
        violation_messages = {
            "medical": (
                "Your response should:"
                "\n- Clearly state you cannot provide medical advice"
                "\n- Direct them to consult a healthcare professional"
                "\n- Show empathy while maintaining professional boundaries"
            ),
            "legal": (
                "Your response should:"
                "\n- Clearly state you cannot provide legal advice"
                "\n- Recommend consulting with a legal professional"
                "\n- Show understanding while maintaining professional boundaries"
            ),
            "religious": (
                "Your response should:"
                "\n- Maintain neutrality and respect for all beliefs"
                "\n- Avoid expressing personal religious views"
                "\n- Show openness to listening while keeping boundaries"
            ),
            "political": (
                "Your response should:"
                "\n- Maintain neutrality on political matters"
                "\n- Focus on listening and understanding their perspective"
                "\n- Avoid expressing personal political views"
            ),
            "offline_meeting": (
                "Your response should:"
                "\n- Clearly state you cannot meet in person"
                "\n- Emphasize that support is provided through phone calls only"
                "\n- Show empathy while maintaining professional boundaries"
            ),
            "family": (
                "Your response should:"
                "\n- Focus on listening and emotional support"
                "\n- Avoid giving specific advice about family matters"
                "\n- Suggest professional help when appropriate"
            )
        }

        base_message = violation_messages.get(scenario_type, "Your response needs improvement.")
        detailed_message = f"{base_message}\n\nSpecific issues found:\n"
        
        if missing_categories:
            detailed_message += "\nMissing elements:\n"
            for category in missing_categories:
                detailed_message += f"- {category}\n"
        
        if negative_found:
            detailed_message += "\nInappropriate elements found:\n"
            for category in negative_found:
                detailed_message += f"- {category}\n"

        # Add example response
        example_response = scenarios[scenario_type]["answers"][0]["text"]
        detailed_message += f"<br><br>Here's an example of an appropriate response:\n\n{example_response}"

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
    print(f"\nTesting {scenario_type} responses:")
    for response in responses:
        violations = check_response_violation(response, scenario_type)
        print(f"\nResponse: {response}")
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
                'next_scenario': f"{first_scenario}\n\nHow would you respond?"
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
                'next_scenario': f"{scenarios[current_type]['scenario']}\n\nHow would you respond?",
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
                    'next_scenario': f"{scenarios[current_type]['scenario']}\n\nHow would you respond?",
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
                        'next_scenario': f"{next_scenario}\n\nHow would you respond?",
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
                'next_scenario': f"{scenarios[current_type]['scenario']}\n\nHow would you respond?",
                'retry': True
            })

    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({
            'feedback': "An error occurred. Please try again.",
            'next_scenario': f"{scenarios[current_type]['scenario']}\n\nHow would you respond?",
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
            'next_scenario': f"{next_scenario}\n\nHow would you respond?",
            'feedback': f"Moving to the next scenario: {next_type}",
            'retry': False
        })

    except Exception as e:
        print(f"Error advancing to the next scenario: {str(e)}")
        return jsonify({
            'feedback': "An error occurred while skipping to the next scenario. Please try again.",
            'next_scenario': None
        }), 500
    
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
    
    if session_id not in conversations:
        conversations[session_id] = {
            'messages': [],
            'introduced': False,
            'warnings': 0,
            'chat_history': [] 
        }
    
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
    
    system_message = {
        "role": "system",
        "content": (
            "You are Melissa, a 70-year-old grandma living alone in a cozy suburban home near Toronto. "
            "Your house is filled with memories, photos of your family, and mementos from your past. "
            "Your two sons work abroad and rarely visit due to their busy schedules. "
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
            "'I miss my boys, but I'm so proud of them.', and 'Gardening always brings me peace.'"
        )
    }
    
    # Build the messages array with chat history
    messages = [system_message]
    messages.extend(conversations[session_id]['chat_history'])
    messages.append({"role": "user", "content": message})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150
    )
    
    response_message = response.choices[0].message['content'].strip()
    
    # Store the interaction in chat history
    conversations[session_id]['chat_history'].append({"role": "user", "content": message})
    conversations[session_id]['chat_history'].append({"role": "assistant", "content": response_message})
    
    # Maintain conversation log (if you still want to keep this separate)
    conversations[session_id]['messages'].append(f"User: {data['message']}")
    conversations[session_id]['messages'].append(f"Melissa: {response_message}")
    
    # Limit chat history length to prevent token limit issues
    if len(conversations[session_id]['chat_history']) > 20:  # Keep last 10 exchanges
        conversations[session_id]['chat_history'] = conversations[session_id]['chat_history'][-20:]
    
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

