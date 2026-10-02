"""Explicitly fictional demo guidance and authored training examples.

These are not Ministry-approved legal sources or real taxpayer complaints.
They exercise software; their metrics do not establish field performance.
"""
GUIDANCE = [
    ("payments", "Payment channels", "en", "Use EVC, ZAAD, EDAHAB, bank or card to make a demo payment. Keep your payment reference and receipt. If a payment is missing, provide its reference to the Revenue Officer; do not pay twice before it is checked."),
    ("filing", "Filing and deadlines", "en", "Submit a tax return by the due date on your filing notice, including a nil return if the notice requires it. For a missed deadline, contact the Revenue Officer to discuss the late return. This demo does not calculate statutory penalties."),
    ("appeals", "Corrections and appeals", "en", "If you disagree with an assessment or a meter reading, request a review and provide the notice, receipt or meter photograph. An officer checks the evidence. AI scores are recommendations, not proof of evasion or a final penalty decision."),
    ("registration", "Registration", "en", "To register a demo business, provide the business name, sector and region through the registration service. A Revenue Officer verifies the application before issuing a taxpayer identifier. Do not upload identity documents to this public demo."),
    ("water", "Water meter readings", "en", "Photograph the meter display clearly and crop out serial numbers. Avoid glare and keep the camera steady. An unreadable or lower reading requires officer review; a lower value can reflect a reset or rollover and does not prove tampering."),
    ("technical", "Portal access", "en", "For a login problem, use the password reset option or contact technical support. Never share passwords, access tokens or payment PINs with an officer or the assistant."),
    ("payments-so", "Bixinta lacagta", "so", "Lacagta tijaabada ah waxaa lagu bixin karaa EVC, ZAAD, EDAHAB, bangi ama kaar. Hayso lambarka tixraaca iyo rasiidka. Haddii lacagta la waayo, la xiriir sarkaalka dakhliga; ha bixin mar labaad inta aan la hubin."),
    ("filing-so", "Gudbinta foomka", "so", "Gudbi foomka canshuurta ka hor taariikhda ku qoran ogeysiiska. Haddii aad dib u dhacdo, la xiriir sarkaalka dakhliga. Tijaabadani ma xisaabiso ganaaxyada sharciga ah."),
    ("appeals-so", "Cabasho iyo dib u eegis", "so", "Haddii aad diidan tahay qiimeynta ama akhriska mitirka, codso dib u eegis oo keen ogeysiiska, rasiidka ama sawirka mitirka. Sarkaal ayaa hubinaya caddeynta. Dhibcaha AI ma aha caddeyn canshuur lunsi ama go'aan ganaax."),
    ("water-so", "Akhriska mitirka biyaha", "so", "Sawir cad ka qaad shaashadda mitirka biyaha. Ka fogow iftiin dhalaalaya oo kamaradda deji. Akhris aan caddayn ama ka hooseeya kii hore wuxuu u baahan yahay dib u eegis sarkaal; tani kaligeed ma caddeyneyso faragelin."),
]


def documents():
    return [{"id": "DEMO-" + key, "title": "RICAP fictional demo guide", "section": title, "language": lang,
             "audience": "PUBLIC", "text": text, "effective_date": "2026-01-01", "superseded": False}
            for key, title, lang, text in GUIDANCE]


# Eight explicit demonstration categories. The assessment does not name the eight.
COMPLAINTS = {
    "payment": ["My payment is missing", "The receipt shows the wrong amount", "I was charged twice", "Bank transfer not credited", "EVC payment failed", "Lacagtii aan bixiyay lama helin", "Rasiidka lacagta ayaa khaldan", "ZAAD lacag ayaan diray but no receipt", "Card charged but balance unchanged", "Refund my duplicate transaction"],
    "filing": ["I cannot submit my tax return", "The filing deadline is incorrect", "My return was rejected", "I need to amend a submitted return", "Nil return submission error", "Foomka canshuurta ma gudbin karo", "Taariikhda gudbinta ayaa khaldan", "Return diiday please help", "The filing form has missing fields", "I submitted the wrong declaration"],
    "registration": ["I need a taxpayer registration number", "My business registration is pending", "The business name is wrong", "Please update my business address", "Close my taxpayer registration", "Diiwaangelinta ganacsiga ayaa daahday", "Magaca ganacsiga waa khaldan", "TIN registration ma helin", "Register a new shop", "My tax identifier is duplicated"],
    "assessment": ["The assessed tax is too high", "I dispute the penalty notice", "Please explain this tax assessment", "My tax bill is incorrect", "I want to appeal the audit decision", "Canshuurta la igu qiimeeyay waa badan tahay", "Ganaaxa ayaan diidanahay", "Assessment wrong waxaan rabaa appeal", "This assessment uses the wrong turnover", "An officer calculated the wrong tax"],
    "water": ["My water meter reading is wrong", "There is no water supply", "The water bill is too high", "My meter is broken", "Water pipe is leaking", "Mitirka biyaha waa jaban yahay", "Biilkii biyaha ayaa khaldan", "Biyo ma jiraan please repair", "Please reconnect my water", "The photographed meter digits are incorrect"],
    "technical": ["I cannot log into the portal", "Password reset email never arrives", "The app keeps crashing", "Website gives a server error", "The portal page is blank", "App-ka ma shaqeynayo", "Furaha sirta ah ma beddeli karo", "Login ma geli karo", "Website is down today", "My account is locked after login"],
    "staff_conduct": ["An officer demanded a bribe", "A staff member was rude", "The officer threatened me", "I want to report corruption", "Staff refused to assist me", "Sarkaal ayaa laaluush iga dalbaday", "Shaqaalaha ayaa i aflagaadeeyay", "Officer rude ayuu ahaa", "An employee asked for unofficial cash", "Report misconduct by a revenue officer"],
    "general": ["Where is the revenue office", "What time do you open", "I need general information", "How can I contact customer service", "Where can I find help", "Xafiiska dakhliga xaggee ku yaal", "Goorma ayuu xafiisku furmaa", "Help ayaan rabaa", "What is your telephone number", "Please explain the available services"],
}

ROUTES = {"payment": "payments-team", "filing": "returns-team", "registration": "registration-team", "assessment": "assessment-review", "water": "water-agency", "technical": "technical-support", "staff_conduct": "integrity-office", "general": "customer-support"}
