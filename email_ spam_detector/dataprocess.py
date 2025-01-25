import random
import csv

# Predefined templates for ham and spam messages
ham_templates = [
    "Hey, are you free to catch up this {}?",
    "Let's meet at {} for {}!",
    "Can you call me back around {}?",
    "Don't forget the {} on {}.",
    "I'll be at the {} if you need me.",
    "See you at the {} {}.",
    "Have you heard about the {}?",
    "How's the {} going?",
    "I was thinking about you when I saw {}.",
    "Let's plan the {} for {}.",
]

spam_templates = [
    "Congratulations! You've won a {}. Click here to claim your prize!",
    "This is a limited offer, {} now and get {} off!",
    "You have been selected for a {}. Act now!",
    "Your account has been flagged. Please visit {} to fix it.",
    "Claim your {} today! Don't miss out.",
    "Earn up to {} per day working from home!",
    "Free {} for the first 100 customers. Don't wait!",
    "Get {} now with our special discount code {}.",
    "Special promotion! Buy one {}, get one free!",
    "Win a free trip to {}! Sign up today.",
]

# Random words to fill in templates
words = [
    "Monday", "meeting", "weekend", "sale", "vacation", "birthday",
    "gift card", "discount", "afternoon", "morning", "evening", "offer",
    "game", "party", "event", "gathering", "conference", "seminar",
    "presentation", "contest", "survey", "promotion", "gift", "trip",
    "bonus", "reward", "watch", "phone", "TV", "laptop", "camera"
]

# Generate ham and spam messages
def generate_message(template, words):
    return template.format(*random.choices(words, k=template.count("{}")))

# Write to CSV file
with open('messages_dataset.csv', 'w', newline='') as csvfile:
    fieldnames = ['v1', 'v2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for _ in range(50000):  # 50,000 ham messages
        writer.writerow({'v1': 'ham', 'v2': generate_message(random.choice(ham_templates), words)})

    for _ in range(50000):  # 50,000 spam messages
        writer.writerow({'v1': 'spam', 'v2': generate_message(random.choice(spam_templates), words)})

print("Dataset with 100,000 entries generated and saved as 'messages_dataset.csv'.")
