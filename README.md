# CompanionLink Training Platform

A Flask-based web application for training volunteers in effective communication with seniors and veterans through simulated chat interactions.

## Overview

This platform provides an interactive training environment where volunteers can practice their communication skills through conversations with AI-powered personas:
- Melissa: A 70-year-old grandmother seeking companionship
- Ian: A 55-year-old veteran adjusting to civilian life

The application features real-time feedback, rapport tracking, and scenario-based training to help volunteers develop appropriate communication strategies.

## Features

### Core Functionality
- Text-based chat interfaces
- Voice chat capability with speech-to-text and text-to-speech
- Real-time conversation analysis and feedback
- Progress tracking and scoring system
- Scenario-based training modules

### Training Scenarios
- Introduction scenarios
- Handling sensitive topics
- Setting appropriate boundaries
- Managing personal information
- Crisis response training

### Rapport Building System
- Dynamic rapport scoring
- Real-time feedback on interaction quality
- Achievement tracking
- Information discovery system
- Progress monitoring

## Technical Requirements

### Dependencies
- Flask
- OpenAI API
- Flask-CORS
- HTTPX
- scikit-learn
- python-dotenv
- NumPy
- SciPy

### Environment Variables
```
OPENAI_API_KEY=your_api_key_here
PORT=10000 (default)
```

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
python app.py
```

## Usage

### Character Selection
- Navigate to `/select` to choose between Melissa or Ian
- Each character provides unique training scenarios and challenges

### Text Chat Training
- `/melissa_chat` - Practice with Melissa
- `/ian_chat` - Practice with Ian
- Real-time feedback on communication style
- Progress tracking and achievement system

### Voice Chat Training
- `/melissa_voicechat` - Voice interaction with Melissa
- Speech-to-text and text-to-speech functionality
- Natural conversation practice

### Guidance System
- `/guidance` - Access training materials
- `/chat_guidance` - Interactive scenario practice
- Real-time violation checking
- Structured feedback on responses

## Scoring System

### Rapport Metrics
- Empathy (0-100)
- Engagement (0-100)
- Flow (0-100)
- Respect (0-100)

### Achievement System
- First Connection
- Patient Listener
- Trust Builder
- Respectful Boundaries
- Empathy Master

## Security Features

- Input validation
- Response filtering
- Sensitive topic management
- Boundary enforcement
- Privacy protection

## Development Guidelines

### Adding New Scenarios
1. Define scenario in the `scenarios` dictionary
2. Include appropriate response templates
3. Set up validation rules
4. Define feedback thresholds

### Modifying Character Behavior
1. Update system messages in respective chat routes
2. Adjust rapport thresholds
3. Modify response patterns
4. Update violation checks

## Error Handling

The application includes comprehensive error handling for:
- Invalid inputs
- API failures
- Session management
- Audio processing
- Response generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description of changes
4. Ensure all tests pass
