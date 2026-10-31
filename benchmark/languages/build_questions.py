"""Write benchmark/languages/{,heldout/}questions.jsonl from the specs below.

Evidence pages are computed from the real PDF text with the grader's own normaliser
(local_agentic_rag.eval.metrics.alias_matches); a fact that is on no page is an error.
"""
import json
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from local_agentic_rag.eval.metrics import alias_matches, normalize_text  # noqa: E402

# (doc, combo, category, q_lang, a_lang, question, reference_answer, facts)
DEV = [
    # ---- hi-qutubminar
    ("hi-qutubminar", "same", "numerical", "hi", "hi", "क़ुतुब मीनार में कितनी सीढ़ियाँ हैं?",
     "क़ुतुब मीनार में ३७९ सीढ़ियाँ हैं।", [["379", "३७९"]]),
    ("hi-qutubminar", "en", "multi_hop", "en", "en",
     "Which ruler built the fifth and final storey of the Qutub Minar, and in which year?",
     "Firoz Shah Tughlaq, in 1368.",
     [["फीरोजशाह", "फ़ीरोज़", "फ़िरोज़", "फिरोज", "फीरोज", "firoz", "feroz", "firuz", "فیروز"], ["1368", "१३६८"]]),
    ("hi-qutubminar", "cross", "factual", "ur", "ur",
     "قطب الدین ایبک کے جانشین کا کیا نام تھا جس نے قطب مینار میں تین منزلیں بڑھائیں؟",
     "ان کا نام التتمش (شمس الدین التتمش) تھا۔",
     [["इल्तुतमिश", "इतुतमिश", "التتمش", "التمش", "iltutmish", "iltutmush", "altamash"]]),
    ("hi-qutubminar", "unanswerable", "unanswerable", "en", "en",
     "How much did it cost to build the Qutub Minar?",
     "The document does not say how much the Qutub Minar cost to build.", []),
    ("hi-qutubminar", "answer-in", "factual", "en", "ur",
     "In which part of Delhi is the Qutub Minar located? Answer in Urdu.",
     "قطب مینار جنوبی دہلی کے علاقے مہرولی میں واقع ہے۔",
     [["महरौली", "مہرولی", "مهرولی", "mehrauli", "mehrouli"]]),
    # ---- mr-raigad
    ("mr-raigad", "same", "numerical", "mr", "mr", "रायगड किल्ल्याला सुमारे किती पायऱ्या आहेत?",
     "रायगड किल्ल्याला सुमारे १४३५ पायऱ्या आहेत.", [["1435", "१४३५"]]),
    ("mr-raigad", "en", "factual", "en", "en", "On what date did Shivaji Maharaj lay siege to Rairi (Raigad)?",
     "On 6 April 1656.", [["1656", "१६५६"], ["april", "एप्रिल", "अप्रैल"]]),
    ("mr-raigad", "cross", "factual", "roman-ur", "roman-ur",
     "Europe ke log Raigad qile ko kis naam se pukarte thay?",
     "Europe ke log use 'Mashriq ka Gibraltar' (Gibraltar of the East) kehte thay.",
     [["जिब्राल्टर", "gibraltar", "jibraltar", "جبرالٹر"]]),
    ("mr-raigad", "answer-in", "numerical", "en", "mr",
     "How high above sea level is Raigad fort, in metres? Answer in Marathi.",
     "रायगड किल्ला समुद्रसपाटीपासून सुमारे ८२० मीटर उंचीवर आहे.", [["820", "८२०"]]),
    # ---- sa-aryabhata
    ("sa-aryabhata", "same", "numerical", "sa", "sa", "आर्यभटः 'आर्यभटीयम्' इति ग्रन्थं कस्मिन् वर्षे अलिखत्?",
     "आर्यभटः क्रि.श. ४९९ तमे वर्षे आर्यभटीयम् अलिखत्।", [["499", "४९९"]]),
    ("sa-aryabhata", "en", "numerical", "en", "en", "In which year was Aryabhata born, according to the text?",
     "In 476 CE.", [["476", "४७६"]]),
    ("sa-aryabhata", "cross", "numerical", "te", "te", "ఆర్యభటీయం గ్రంథంలో ఎన్ని పద్యాలు ఉన్నాయి?",
     "ఆర్యభటీయంలో 108 (నూట ఎనిమిది) పద్యాలు ఉన్నాయి.",
     [["108", "१०८", "౧౦౮", "अष्टाधिकशत", "నూట ఎనిమిది", "one hundred and eight", "one hundred eight"]]),
    ("sa-aryabhata", "unanswerable", "unanswerable", "hi", "hi", "आर्यभट की मृत्यु किस नगर में हुई थी?",
     "दस्तावेज़ में आर्यभट की मृत्यु के स्थान का उल्लेख नहीं है।", []),
    # ---- ne-sagarmatha
    ("ne-sagarmatha", "same", "numerical", "ne", "ne", "सगरमाथाको उचाइ समुद्र सतहबाट कति मिटर छ?",
     "सगरमाथाको उचाइ समुद्र सतहबाट ८,८४८.८६ मिटर छ।",
     [["8848.86", "८८४८.८६", "8,848.86", "८,८४८.८६"]]),
    ("ne-sagarmatha", "en", "multi_hop", "en", "en", "Who first stood on the summit of Sagarmatha, and in which year?",
     "Tenzing Norgay Sherpa of Nepal and Sir Edmund Hillary of New Zealand, on 29 May 1953.",
     [["तेन्जिङ", "नोर्गे", "tenzing", "norgay"], ["हिलारी", "hillary"], ["1953", "१९५३"]]),
    ("ne-sagarmatha", "cross", "factual", "bn", "bn", "সগরমাথা নামটি কে রেখেছিলেন?",
     "ইতিহাস শিরোমণি বাবুরাম আচার্য সগরমাথা নামটি রেখেছিলেন।",
     [["बाबुराम", "বাবুরাম", "baburam", "babu ram"]]),
    # ---- bh-bhikharithakur
    ("bh-bhikharithakur", "same", "factual", "bh", "bh", "भिखारी ठाकुर के जनम कब भइल रहे?",
     "भिखारी ठाकुर के जनम 18 दिसंबर 1887 के भइल रहे।",
     [["1887", "१८८७"], ["दिसंबर", "दिसम्बर", "december"]]),
    ("bh-bhikharithakur", "en", "factual", "en", "en", "What is the name of Bhikhari Thakur's most famous play?",
     "Bidesiya.", [["बिदेसिया", "bidesiya", "bidesia", "videsiya", "bideshiya"]]),
    ("bh-bhikharithakur", "cross", "multi_hop", "hi", "hi",
     "भिखारी ठाकुर का निधन किस वर्ष हुआ और उस समय उनकी आयु कितनी थी?",
     "भिखारी ठाकुर का निधन 10 जुलाई 1971 को 83 वर्ष की आयु में हुआ।", [["1971", "१९७१"], ["83", "८३"]]),
    ("bh-bhikharithakur", "unanswerable", "unanswerable", "mr", "mr", "भिखारी ठाकुर यांना किती मुले होती?",
     "दस्तऐवजात भिखारी ठाकुर यांच्या मुलांबद्दल माहिती नाही.", []),
    # ---- awa-tulsidas
    ("awa-tulsidas", "same", "numerical", "awa", "awa", "तुलसीदास अपने जीवनकाल मा कुल केतने ग्रन्थन के रचना कीन्हेनि?",
     "तुलसीदास अपने जीवनकाल मा कुल १२ ग्रन्थन के रचना कीन्हेनि।", [["12", "१२"]]),
    ("awa-tulsidas", "en", "factual", "en", "en", "What was the name of Tulsidas's wife?",
     "Ratnavali, the daughter of Deenbandhu Pathak.", [["रत्नावली", "ratnavali", "ratnawali"]]),
    ("awa-tulsidas", "cross", "factual", "te", "te", "తులసీదాస్‌కు చిన్నప్పుడు ఇంట్లో పెట్టిన పేరు ఏమిటి?",
     "రామ్‌బోలా (रामबोला).",
     [["रामबोला", "rambola", "ram bola", "రామ్‌బోలా", "రామ్బోలా", "రాంబోలా", "రామబోలా"]]),
    ("awa-tulsidas", "answer-in", "numerical", "hi", "ta",
     "तुलसीदास ने कितने वर्ष की उम्र में सोरों को हमेशा के लिए छोड़ दिया? उत्तर तमिल में दीजिए।",
     "துளசிதாசர் 36 வயதில் சோரோனை நிரந்தரமாக விட்டுச் சென்றார்.", [["36", "३६", "௩௬", "முப்பத்தாறு"]]),
    # ---- mai-vidyapati
    ("mai-vidyapati", "same", "factual", "mai", "mai", "विद्यापतिक पिताक नाम की छल?",
     "विद्यापतिक पिताक नाम गणपति ठाकुर छल।", [["गणपति", "ganapati", "ganpati"]]),
    ("mai-vidyapati", "en", "factual", "en", "en", "In which village was Vidyapati born?",
     "In Bisfi village, in present-day Madhubani district of Bihar.",
     [["बिस्फी", "बिसफी", "bisfi", "bisphi", "bispi"]]),
    ("mai-vidyapati", "cross", "numerical", "ne", "ne", "विद्यापतिको जन्म कुन सालमा भएको थियो?",
     "विद्यापतिको जन्म सन् १३५२ मा भएको थियो।", [["1352", "१३५२"]]),
    ("mai-vidyapati", "unanswerable", "unanswerable", "ur", "ur", "ودیاپتی کی بیوی کا نام کیا تھا؟",
     "دستاویز میں ودیاپتی کی بیوی کے نام کا ذکر نہیں ہے۔", []),
    # ---- ur-badshahimosque
    ("ur-badshahimosque", "same", "numerical", "ur", "ur", "بادشاہی مسجد میں بیک وقت کتنے لوگ نماز ادا کر سکتے ہیں؟",
     "بادشاہی مسجد میں بیک وقت 60 ہزار لوگ نماز ادا کر سکتے ہیں۔",
     [["60 ہزار", "60000", "60,000", "٦٠ ہزار", "۶۰ ہزار", "ساٹھ ہزار", "sixty thousand", "60 thousand", "60 hazar"]]),
    ("ur-badshahimosque", "en", "multi_hop", "en", "en",
     "Which Mughal emperor built the Badshahi Mosque, and in which year was it completed?",
     "Aurangzeb Alamgir; it was completed in 1673.",
     [["aurangzeb", "اورنگزیب", "औरंगज़ेब", "औरंगजेब", "alamgir", "عالمگیر"], ["1673", "۱۶۷۳", "١٦٧٣"]]),
    ("ur-badshahimosque", "cross", "numerical", "ar", "ar", "في أي عام اكتمل الترميم الكامل لمسجد بادشاهي في لاهور؟",
     "اكتمل الترميم الكامل للمسجد في عام 1960.", [["1960", "١٩٦٠", "۱۹۶۰"]]),
    # ---- bn-howrahbridge
    ("bn-howrahbridge", "same", "numerical", "bn", "bn", "হাওড়া সেতু দিয়ে দৈনিক কতগুলি যানবাহন চলাচল করে?",
     "হাওড়া সেতু দিয়ে দৈনিক প্রায় ৮০,০০০ যানবাহন চলাচল করে।",
     [["80000", "৮০,০০০", "80,000", "৮০ হাজার", "80 হাজার", "আশি হাজার", "80 thousand", "eighty thousand"]]),
    ("bn-howrahbridge", "en", "factual", "en", "en", "Who designed the old (pontoon) Howrah Bridge?",
     "Bradford Leslie.", [["leslie", "লেসলি"]]),
    ("bn-howrahbridge", "cross", "numerical", "ta", "ta",
     "முதல் மிதக்கும் ஹவுரா பாலம் எந்த ஆண்டு போக்குவரத்துக்குத் திறக்கப்பட்டது?",
     "முதல் மிதக்கும் ஹவுரா பாலம் 1874 ஆம் ஆண்டு போக்குவரத்துக்குத் திறக்கப்பட்டது.",
     [["1874", "১৮৭৪", "௧௮௭௪"]]),
    ("bn-howrahbridge", "unanswerable", "unanswerable", "bn", "bn", "হাওড়া সেতু নির্মাণে মোট কত টাকা খরচ হয়েছিল?",
     "নথিতে হাওড়া সেতু নির্মাণের মোট খরচের উল্লেখ নেই।", []),
    ("bn-howrahbridge", "answer-in", "numerical", "en", "bn",
     "In which year was the Howrah Bridge renamed Rabindra Setu? Answer in Bengali.",
     "১৯৬৫ সালের ১৪ জুন সেতুটির নাম পরিবর্তন করে রবীন্দ্র সেতু রাখা হয়।", [["1965", "১৯৬৫"]]),
    # ---- te-charminar
    ("te-charminar", "same", "numerical", "te", "te", "చార్మినార్‌ను ఏ సంవత్సరంలో నిర్మించారు?",
     "చార్మినార్‌ను 1591లో ముహమ్మద్ కులీ కుతుబ్ షా నిర్మించాడు.", [["1591", "౧౫౯౧"]]),
    ("te-charminar", "en", "multi_hop", "en", "en",
     "Which ruler of the Qutb Shahi dynasty built the Charminar, and in which year?",
     "Muhammad Quli Qutb Shah, the fifth Qutb Shahi ruler, built it in 1591.",
     [["కులీ", "quli", "qoli", "kuli"], ["1591", "౧౫౯౧"]]),
    ("te-charminar", "cross", "numerical", "mr", "mr", "चारमिनारच्या चारही बाजूंना कमानी कोणत्या वर्षी बांधल्या गेल्या?",
     "चारमिनारच्या चारही बाजूंना कमानी १५९२ मध्ये बांधल्या गेल्या.", [["1592", "१५९२", "౧౫౯౨"]]),
    ("te-charminar", "answer-in", "numerical", "te", "hi",
     "చార్మినార్ చుట్టూ నిర్మించిన నాలుగు కమాన్‌ల ఎత్తు ఎంత? సమాధానం హిందీలో ఇవ్వండి.",
     "चारमीनार के चारों ओर बनी कमानें 60 फ़ुट ऊँची (और 30 फ़ुट चौड़ी) हैं।", [["60", "౬౦", "६०", "साठ"]]),
    # ---- ta-brihadeeswarar
    ("ta-brihadeeswarar", "same", "factual", "ta", "ta", "தஞ்சைப் பெருவுடையார் கோயிலைக் கட்டுவித்த சோழப் பேரரசர் யார்?",
     "முதலாம் இராசராச சோழன்.", [["இராசராச", "ராஜராஜ", "rajaraja", "raja raja", "rajarajan"]]),
    ("ta-brihadeeswarar", "en", "numerical", "en", "en",
     "In which year was the construction of the Brihadeeswarar (Peruvudaiyar) temple completed?",
     "In 1010 CE.", [["1010"]]),
    ("ta-brihadeeswarar", "cross", "numerical", "hi", "hi", "तंजावुर का बृहदेश्वर मंदिर चेन्नई से लगभग कितने किलोमीटर दूर है?",
     "बृहदेश्वर मंदिर चेन्नई से लगभग 350 किलोमीटर दूर है।", [["350", "३५०"]]),
    ("ta-brihadeeswarar", "unanswerable", "unanswerable", "ta", "ta", "இக்கோயிலின் விமானத்தின் உயரம் எத்தனை மீட்டர்?",
     "விமானத்தின் உயரம் ஆவணத்தில் குறிப்பிடப்படவில்லை.", []),
    ("ta-brihadeeswarar", "answer-in", "numerical", "hi", "en",
     "बृहदेश्वर मंदिर को यूनेस्को ने किस वर्ष विश्व धरोहर घोषित किया? उत्तर अंग्रेज़ी में दीजिए।",
     "UNESCO declared it a World Heritage Site in 1987.", [["1987"]]),
    # ---- kn-mysorepalace
    ("kn-mysorepalace", "same", "multi_hop", "kn", "kn",
     "ಈಗಿನ ಮೈಸೂರು ಅರಮನೆಯ ನಿರ್ಮಾಣ ಯಾವ ವರ್ಷದಲ್ಲಿ ಪ್ರಾರಂಭವಾಯಿತು ಮತ್ತು ಯಾವಾಗ ಮುಗಿಯಿತು?",
     "ನಿರ್ಮಾಣ ೧೮೯೭ ರಲ್ಲಿ ಪ್ರಾರಂಭವಾಗಿ ೧೯೧೨ ರಲ್ಲಿ ಮುಗಿಯಿತು.", [["1897", "೧೮೯೭"], ["1912", "೧೯೧೨"]]),
    ("kn-mysorepalace", "en", "factual", "en", "en", "Before Mysore, which town was the capital of the Mysore kingdom?",
     "Srirangapatna.",
     [["ಶ್ರೀರಂಗಪಟ್ಟಣ", "srirangapatna", "srirangapattana", "seringapatam", "shrirangapattana"]]),
    ("kn-mysorepalace", "cross", "numerical", "hinglish", "hinglish",
     "Tipu Sultan ki maut kis saal hui thi, jiske baad Mysore ke raja Srirangapatna se Mysore aa gaye?",
     "Tipu Sultan ki maut 1799 mein hui thi.", [["1799", "೧೭೯೯"]]),
    ("kn-mysorepalace", "answer-in", "factual", "en", "te",
     "Which British engineer was appointed to build the new Mysore Palace? Answer in Telugu.",
     "కొత్త అరమనను నిర్మించడానికి బ్రిటిష్ ఇంజనీర్ హెన్రీ ఇర్విన్‌ను నియమించారు.",
     [["ಇರ್ವಿನ್", "irwin", "ఇర్విన్", "इर्विन"]]),
    # ---- ml-padmanabhaswamy
    ("ml-padmanabhaswamy", "same", "factual", "ml", "ml",
     "തൃപ്പടിദാനം വഴി രാജ്യം ശ്രീപത്മനാഭസ്വാമിക്ക് സമർപ്പിച്ച തിരുവിതാംകൂർ രാജാവ് ആര്?",
     "അനിഴം തിരുനാൾ മാർത്താണ്ഡവർമ്മ.",
     [["മാർത്താണ്ഡവർമ്മ", "മാർത്താണ്ഡ വർമ്മ", "marthanda varma", "marthandavarma", "marthanda"]]),
    ("ml-padmanabhaswamy", "en", "numerical", "en", "en",
     "How many days does each of the temple's two main annual festivals last?",
     "Ten days each.", [["10", "ten", "പത്ത്", "പത്തു"]]),
    ("ml-padmanabhaswamy", "cross", "factual", "kn", "kn", "ಈ ದೇವಾಲಯದ ಪ್ರಧಾನ ಗೋಪುರವನ್ನು ಯಾವ ಶೈಲಿಯಲ್ಲಿ ನಿರ್ಮಿಸಲಾಗಿದೆ?",
     "ದ್ರಾವಿಡ ಶೈಲಿಯಲ್ಲಿ.", [["ദ്രാവിഡ", "ದ್ರಾವಿಡ", "dravida", "dravidian"]]),
    ("ml-padmanabhaswamy", "unanswerable", "unanswerable", "en", "en",
     "In which year was the main gopuram of the Padmanabhaswamy Temple completed?",
     "The document does not give the year the main gopuram was completed.", []),
    # ---- gu-statueofunity
    ("gu-statueofunity", "same", "numerical", "gu", "gu", "સ્ટેચ્યુ ઓફ યુનિટીની કુલ ઊંચાઈ કેટલી છે?",
     "સ્ટેચ્યુ ઓફ યુનિટીની કુલ ઊંચાઈ ૧૮૨ મીટર છે.", [["182", "૧૮૨"]]),
    ("gu-statueofunity", "en", "multi_hop", "en", "en",
     "Who designed the Statue of Unity, and who formally inaugurated it on 31 October 2018?",
     "Ram V. Sutar designed it; Prime Minister Narendra Modi inaugurated it.",
     [["સુથાર", "sutar"], ["મોદી", "modi"]]),
    ("gu-statueofunity", "cross", "factual", "hinglish", "hinglish", "Statue of Unity ka construction kab shuru hua tha?",
     "Construction 31 October 2013 ko shuru hua tha.",
     [["2013", "૨૦૧૩"], ["october", "ઓક્ટોબર", "oktober", "aktubar", "अक्टूबर"]]),
    # ---- pa-harmandirsahib
    ("pa-harmandirsahib", "same", "multi_hop", "pa", "pa", "ਹਰਿਮੰਦਰ ਸਾਹਿਬ ਦੀ ਨੀਂਹ ਕਦੋਂ ਅਤੇ ਕਿਸ ਨੇ ਰੱਖੀ ਸੀ?",
     "ਨੀਂਹ 28 ਦਸੰਬਰ 1588 ਨੂੰ ਸਾਈਂ ਮੀਆਂ ਮੀਰ ਜੀ ਨੇ ਰੱਖੀ ਸੀ।",
     [["1588", "੧੫੮੮"], ["ਮੀਆਂ ਮੀਰ", "mian mir", "miyan mir", "mian meer", "میاں میر"]]),
    ("pa-harmandirsahib", "en", "numerical", "en", "en",
     "In which year did Guru Arjan Dev install a manuscript of the Adi Granth at Harmandir Sahib?",
     "In 1604.", [["1604", "੧੬੦੪"]]),
    ("pa-harmandirsahib", "cross", "factual", "gu", "gu", "હરિમંદિર સાહિબ પર સોનાનું પડ કોણે ચઢાવ્યું હતું?",
     "મહારાજા રણજીત સિંહે 1830ના દાયકામાં સોનાનું પડ ચઢાવ્યું હતું.",
     [["ਰਣਜੀਤ ਸਿੰਘ", "ranjit singh", "રણજીત સિંહ", "રણજીતસિંહ", "રણજિત સિંહ", "રણજીત સિંગ", "रणजीत सिंह"]]),
    ("pa-harmandirsahib", "unanswerable", "unanswerable", "hi", "hi", "हरमंदिर साहिब के सरोवर की गहराई कितनी है?",
     "दस्तावेज़ में सरोवर की गहराई नहीं दी गई है।", []),
    # ---- or-konark
    ("or-konark", "same", "numerical", "or", "or", "କୋଣାର୍କ ମନ୍ଦିରର ଚଟାଣ ଚାରିପଟେ କେତୋଟି ଚକ ଅଛି?",
     "ଚଟାଣ ଚାରିପଟେ ୨୪ଟି ଚକ ଅଛି ।", [["24", "୨୪"]]),
    ("or-konark", "en", "factual", "en", "en", "Which king is said to have built the Konark Sun Temple around 1250 CE?",
     "King Langula Narasimha Deva of the Ganga dynasty.",
     [["ନରସିଂହ", "narasimha", "narasingha", "narsingh", "narasinga", "narasinha"]]),
    ("or-konark", "cross", "factual", "bn", "bn", "পশ্চিমী নাবিকরা কোণার্ক মন্দিরকে কী নামে ডাকত?",
     "তারা একে কৃষ্ণ প্যাগোডা (ব্ল্যাক প্যাগোডা) বলত।",
     [["କୃଷ୍ଣ ପାଗୋଡ", "black pagoda", "কৃষ্ণ প্যাগোডা", "কালো প্যাগোডা", "ব্ল্যাক প্যাগোডা", "krishna pagoda"]]),
    # ---- en-goldengatebridge
    ("en-goldengatebridge", "same", "factual", "en", "en", "On what date did the Golden Gate Bridge open to the public?",
     "On May 27, 1937.", [["1937"], ["may 27", "27 may", "may 27th", "27th may"]]),
    ("en-goldengatebridge", "cross", "factual", "hi", "hi", "गोल्डन गेट ब्रिज परियोजना के मुख्य इंजीनियर कौन थे?",
     "जोसेफ़ स्ट्रॉस (Joseph Strauss)।", [["strauss", "स्ट्रॉस", "स्ट्रास", "स्ट्राउस"]]),
    ("en-goldengatebridge", "unanswerable", "unanswerable", "fr", "fr", "Combien a coûté la construction du Golden Gate Bridge ?",
     "Le document n'indique pas le coût de construction du pont.", []),
    ("en-goldengatebridge", "answer-in", "numerical", "en", "fr",
     "How long is the main span of the Golden Gate Bridge, in metres? Answer in French.",
     "La travée principale mesure 1 280 mètres (4 200 pieds).", [["1280", "1 280", "1,280", "1.280"]]),
    # ---- fr-toureiffel
    ("fr-toureiffel", "same", "numerical", "fr", "fr", "Quelle était la hauteur de la tour Eiffel à l'origine ?",
     "À l'origine, la tour Eiffel mesurait 312 mètres.", [["312"]]),
    ("fr-toureiffel", "en", "factual", "en", "en", "Which two engineers of Eiffel's company conceived the Eiffel Tower?",
     "Maurice Koechlin and Émile Nouguier.", [["koechlin"], ["nouguier"]]),
    ("fr-toureiffel", "cross", "numerical", "ru", "ru", "Сколько посетителей приняла Эйфелева башня с момента открытия для публики?",
     "Более 300 миллионов посетителей.",
     [["300 millions", "300 миллионов", "300 млн", "300 million", "300000000", "300 000 000", "триста миллионов"]]),
    # ---- zh-threegorgesdam
    ("zh-threegorgesdam", "same", "numerical", "zh", "zh", "三峡水电站共安装了多少台70万千瓦的水轮发电机组？",
     "共安装了32台70万千瓦水轮发电机组。", [["32", "三十二"]]),
    ("zh-threegorgesdam", "en", "factual", "en", "en",
     "Which American engineer, who had led work on the Hoover Dam, surveyed the Three Gorges in 1944?",
     "John L. Savage.", [["savage", "萨凡奇"]]),
    ("zh-threegorgesdam", "cross", "multi_hop", "hi", "hi",
     "तीन घाटियों (Three Gorges) में बाँध बनाने का विचार सबसे पहले किसकी पुस्तक में मिलता है, और वह पुस्तक किस वर्ष प्रकाशित हुई?",
     "सुन यात-सेन (孙中山) की पुस्तक 'राष्ट्र निर्माण की योजना' (建国方略) में, जो 1919 में प्रकाशित हुई।",
     [["孙中山", "यात-सेन", "यात सेन", "यातसेन", "sun yat-sen", "sun yat sen", "sun zhongshan"], ["1919", "१९१९"]]),
    ("zh-threegorgesdam", "unanswerable", "unanswerable", "zh", "zh", "三峡大坝的建设总投资是多少元？",
     "文中没有给出三峡大坝的建设总投资。", []),
    # ---- ja-tokyoskytree
    ("ja-tokyoskytree", "same", "numerical", "ja", "ja", "東京スカイツリーの高さは何メートルですか？",
     "高さは634メートルです。", [["634", "六百三十四"]]),
    ("ja-tokyoskytree", "en", "multi_hop", "en", "en", "Which company built Tokyo Skytree and which firm designed it?",
     "Obayashi Corporation built it; Nikken Sekkei designed it.", [["大林組", "obayashi"], ["日建設計", "nikken"]]),
    ("ja-tokyoskytree", "cross", "factual", "zh", "zh", "东京晴空塔是哪一天竣工的？",
     "东京晴空塔于2012年2月29日竣工。", [["2月29日", "29 february", "february 29", "2012-02-29"]]),
    # ---- ru-baikal
    ("ru-baikal", "same", "numerical", "ru", "ru", "Какова максимальная глубина озера Байкал?",
     "Максимальная глубина Байкала — 1642 метра.", [["1642", "1 642"]]),
    ("ru-baikal", "en", "factual", "en", "en", "Which river flows out of Lake Baikal?",
     "The Angara.", [["ангар", "angara"]]),
    ("ru-baikal", "cross", "numerical", "ja", "ja", "バイカル湖の湖岸線の長さは何キロメートルですか？",
     "湖岸線の長さは2000キロメートルです。", [["2000", "2 000", "2,000", "二千"]]),
    ("ru-baikal", "unanswerable", "unanswerable", "ru", "ru", "Сколько видов рыб обитает в Байкале?",
     "В документе не указано, сколько видов рыб обитает в Байкале.", []),
]

# library variants of each doc's `same` question: (doc, question)
LIBRARY = {
    "hi-qutubminar": "दिल्ली के महरौली में स्थित क़ुतुब मीनार में कुल कितनी सीढ़ियाँ हैं?",
    "mr-raigad": "महाराष्ट्रातील रायगड किल्ल्याला सुमारे किती पायऱ्या आहेत?",
    "sa-aryabhata": "गणितज्ञः आर्यभटः 'आर्यभटीयम्' इति ग्रन्थं कस्मिन् वर्षे अलिखत्?",
    "ne-sagarmatha": "संसारको सबैभन्दा अग्लो चुचुरो सगरमाथाको उचाइ कति मिटर छ?",
    "bh-bhikharithakur": "भोजपुरी के शेक्सपीयर कहाए वाला भिखारी ठाकुर के जनम कब भइल रहे?",
    "awa-tulsidas": "संत-कवि तुलसीदास अपने जीवनकाल मा कुल केतने ग्रन्थन के रचना कीन्हेनि?",
    "mai-vidyapati": "मैथिल कवि कोकिल विद्यापतिक पिताक नाम की छल?",
    "ur-badshahimosque": "لاہور کی بادشاہی مسجد میں بیک وقت کتنے لوگ نماز ادا کر سکتے ہیں؟",
    "bn-howrahbridge": "কলকাতার হাওড়া সেতু (রবীন্দ্র সেতু) দিয়ে দৈনিক কতগুলি যানবাহন চলাচল করে?",
    "te-charminar": "హైదరాబాదులోని చార్మినార్‌ను ఏ సంవత్సరంలో నిర్మించారు?",
    "ta-brihadeeswarar": "தஞ்சாவூரில் உள்ள பெருவுடையார் கோயிலைக் கட்டுவித்த சோழப் பேரரசர் யார்?",
    "kn-mysorepalace": "ಮೈಸೂರಿನ ಅಂಬಾ ವಿಲಾಸ ಅರಮನೆಯ ನಿರ್ಮಾಣ ಯಾವ ವರ್ಷದಲ್ಲಿ ಪ್ರಾರಂಭವಾಯಿತು ಮತ್ತು ಯಾವಾಗ ಮುಗಿಯಿತು?",
    "ml-padmanabhaswamy": "തിരുവനന്തപുരത്തെ ശ്രീപത്മനാഭസ്വാമി ക്ഷേത്രത്തിന് തൃപ്പടിദാനം വഴി രാജ്യം സമർപ്പിച്ച തിരുവിതാംകൂർ രാജാവ് ആര്?",
    "gu-statueofunity": "વલ્લભભાઈ પટેલને સમર્પિત સ્ટેચ્યુ ઓફ યુનિટીની કુલ ઊંચાઈ કેટલી છે?",
    "pa-harmandirsahib": "ਅੰਮ੍ਰਿਤਸਰ ਦੇ ਹਰਿਮੰਦਰ ਸਾਹਿਬ ਦੀ ਨੀਂਹ ਕਦੋਂ ਅਤੇ ਕਿਸ ਨੇ ਰੱਖੀ ਸੀ?",
    "or-konark": "ଓଡ଼ିଶାର କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିରର ଚଟାଣ ଚାରିପଟେ କେତୋଟି ଚକ ଅଛି?",
    "en-goldengatebridge": "On what date did San Francisco's Golden Gate Bridge open to the public?",
    "fr-toureiffel": "Quelle était la hauteur de la tour Eiffel de Paris à l'origine ?",
    "zh-threegorgesdam": "长江三峡大坝的水电站共安装了多少台70万千瓦的水轮发电机组？",
    "ja-tokyoskytree": "東京都墨田区にある東京スカイツリーの高さは何メートルですか？",
    "ru-baikal": "Какова максимальная глубина озера Байкал в Восточной Сибири?",
}

# two-book questions: (docs, category, q_lang, question, answer, facts)
TWO_BOOKS = [
    (["te-charminar", "ur-badshahimosque"], "comparison", "en",
     "Which was built first, the Charminar in Hyderabad or the Badshahi Mosque in Lahore? Give the year for each.",
     "The Charminar (1591) was built first; the Badshahi Mosque was built in 1671-1673.",
     [["1591", "౧౫౯౧"], ["1673", "1671", "۱۶۷۳", "١٦٧٣"]]),
    (["ja-tokyoskytree", "gu-statueofunity"], "multi_hop", "en",
     "Give the height of Tokyo Skytree and the total height of the Statue of Unity, in metres.",
     "Tokyo Skytree is 634 m tall; the Statue of Unity is 182 m tall.", [["634"], ["182", "૧૮૨"]]),
    (["hi-qutubminar", "mr-raigad"], "multi_hop", "hi",
     "क़ुतुब मीनार में कितनी सीढ़ियाँ हैं, और रायगड किले में लगभग कितनी सीढ़ियाँ हैं?",
     "क़ुतुब मीनार में ३७९ सीढ़ियाँ हैं और रायगड किले में लगभग १४३५ सीढ़ियाँ हैं।",
     [["379", "३७९"], ["1435", "१४३५"]]),
    (["te-charminar", "ta-brihadeeswarar"], "multi_hop", "te",
     "చార్మినార్‌ను ఏ సంవత్సరంలో నిర్మించారు, తంజావూరు బృహదీశ్వరాలయం నిర్మాణం ఏ సంవత్సరంలో పూర్తయింది?",
     "చార్మినార్ 1591లో నిర్మించబడింది; తంజావూరు బృహదీశ్వరాలయం నిర్మాణం 1010లో పూర్తయింది.",
     [["1591", "౧౫౯౧"], ["1010", "౧౦౧౦"]]),
    (["en-goldengatebridge", "ru-baikal"], "multi_hop", "fr",
     "En quelle année le Golden Gate Bridge a-t-il été ouvert au public, et quelle est la profondeur maximale du lac Baïkal ?",
     "Le Golden Gate Bridge a été ouvert en 1937 ; la profondeur maximale du lac Baïkal est de 1 642 mètres.",
     [["1937"], ["1642", "1 642"]]),
    (["ur-badshahimosque", "bn-howrahbridge"], "multi_hop", "bn",
     "লাহোরের বাদশাহী মসজিদে একসঙ্গে কতজন নামাজ পড়তে পারেন, এবং হাওড়া সেতু দিয়ে দৈনিক কতগুলি যানবাহন চলাচল করে?",
     "বাদশাহী মসজিদে একসঙ্গে ৬০,০০০ (৬০ হাজার) জন নামাজ পড়তে পারেন; হাওড়া সেতু দিয়ে দৈনিক প্রায় ৮০,০০০ যানবাহন চলাচল করে।",
     [["60 ہزار", "60000", "৬০,০০০", "৬০ হাজার", "60 হাজার", "ষাট হাজার", "60 thousand"],
      ["80000", "৮০,০০০", "৮০ হাজার", "80 হাজার", "আশি হাজার", "80 thousand"]]),
]

HELDOUT = [
    ("bh-patna", "same", "factual", "bh", "bh", "पटना क प्राचीन नाँव का रहे?", "पटना क प्राचीन नाँव पाटलिपुत्र रहे।",
     [["पाटलिपुत्र", "पाटलीपुत्र", "pataliputra", "patliputra", "pataliputr"]]),
    ("bh-patna", "en", "factual", "en", "en", "Which Sikh guru was born in Patna?",
     "Guru Gobind Singh, the tenth and last Sikh guru.", [["गोबिंद", "gobind", "govind"]]),
    ("bh-patna", "cross", "numerical", "mr", "mr", "२०११ मध्ये पटना हे भारतातील कितव्या क्रमांकाचे सर्वात मोठे शहर होते?",
     "२०११ मध्ये पटना हे भारतातील १९ वे सर्वात मोठे शहर होते.", [["19", "१९"]]),
    ("fr-viaducdemillau", "same", "numerical", "fr", "fr", "Combien a coûté le viaduc de Millau ?",
     "Le viaduc de Millau a coûté 320 millions d'euros.", [["320"]]),
    ("fr-viaducdemillau", "en", "factual", "en", "en", "Which company financed and built the Millau Viaduct?",
     "The Eiffage group.", [["eiffage"]]),
    ("fr-viaducdemillau", "cross", "numerical", "hi", "hi", "मिलाउ वायाडक्ट का उद्घाटन किस वर्ष हुआ था?",
     "मिलाउ वायाडक्ट का उद्घाटन 14 दिसंबर 2004 को हुआ था।", [["2004", "२००४"]]),
    ("sa-kalidasa", "same", "factual", "sa", "sa", "किंवदन्त्यनुसारं कालिदासः कस्य नवरत्नेषु अन्यतमः आसीत्?",
     "कालिदासः विक्रमादित्यस्य नवरत्नेषु अन्यतमः आसीत्।", [["विक्रमादित्य", "vikramaditya"]]),
    ("sa-kalidasa", "en", "multi_hop", "en", "en", "Name the two mahakavyas (great epic poems) attributed to Kalidasa.",
     "Raghuvamsha and Kumarasambhava.",
     [["रघुवंश", "raghuvamsha", "raghuvansha", "raghuvamsa", "raghuvansh"],
      ["कुमारसम्भव", "kumarasambhava", "kumarsambhav", "kumarasambhav", "kumara sambhava"]]),
    ("sa-kalidasa", "cross", "factual", "te", "te", "కాళిదాసు రచించిన ఖండకావ్యం పేరు ఏమిటి?", "మేఘదూతం (మేఘసందేశం).",
     [["मेघदूत", "meghaduta", "meghdoot", "meghadoot", "meghdut", "మేఘదూత", "మేఘసందేశ"]]),
    ("ta-kallanai", "same", "numerical", "ta", "ta", "கல்லணையின் நீளம் எவ்வளவு?", "கல்லணையின் நீளம் 1080 அடி.",
     [["1080", "1,080"]]),
    ("ta-kallanai", "en", "factual", "en", "en", "Which Chola king built the Kallanai dam?", "Karikala Chola (Karikalan).",
     [["கரிகால", "karikala", "karikalan", "karikal"]]),
    ("ta-kallanai", "cross", "numerical", "kn", "kn", "ಕಲ್ಲಣೆ ಅಣೆಕಟ್ಟಿನ ಮೇಲೆ ಸೇತುವೆಯನ್ನು ಯಾವ ವರ್ಷದಲ್ಲಿ ಕಟ್ಟಲಾಯಿತು?",
     "ಸೇತುವೆಯನ್ನು ೧೮೩೯ ರಲ್ಲಿ ಕಟ್ಟಲಾಯಿತು.", [["1839", "೧೮೩೯"]]),
    ("te-golconda", "same", "numerical", "te", "te", "గోల్కొండ కోట హైదరాబాదుకు ఎంత దూరంలో ఉంది?",
     "గోల్కొండ కోట హైదరాబాదుకు 16 కి.మీ. దూరంలో ఉంది.", [["16", "౧౬"]]),
    ("te-golconda", "en", "multi_hop", "en", "en",
     "Which chief minister first held Independence Day celebrations at Golconda Fort, and in which year?",
     "Kalvakuntla Chandrashekar Rao, on 15 August 2014.",
     [["చంద్రశేఖర్", "chandrashekar", "chandrasekhar", "chandrashekhar", "chandra shekar", "kcr"], ["2014", "౨౦౧౪"]]),
    ("te-golconda", "cross", "numerical", "ur", "ur", "گولکنڈہ قلعہ کتنی اونچی پہاڑی پر بنایا گیا ہے؟",
     "گولکنڈہ قلعہ 120 میٹر اونچی پہاڑی پر بنایا گیا ہے۔", [["120", "۱۲۰", "١٢٠"]]),
    ("ur-minarepakistan", "same", "numerical", "ur", "ur", "مینار پاکستان کی تعمیر کب مکمل ہوئی؟",
     "مینار پاکستان کی تعمیر 21 اکتوبر 1968ء کو مکمل ہوئی۔", [["1968", "۱۹۶۸", "١٩٦٨"]]),
    ("ur-minarepakistan", "en", "factual", "en", "en", "Who designed the Minar-e-Pakistan?",
     "The Turkish architect Nasreddin Murat-Khan.", [["مراد خان", "murat khan", "murat-khan", "murad khan", "muratkhan"]]),
    ("ur-minarepakistan", "cross", "numerical", "hi", "hi", "मीनार-ए-पाकिस्तान के निर्माण की कुल लागत कितनी थी?",
     "मीनार-ए-पाकिस्तान के निर्माण की कुल लागत 75 लाख रुपये थी।",
     [["75 لاکھ", "75 लाख", "७५ लाख", "75 lakh", "7.5 million", "7500000", "पचहत्तर लाख"]]),
]
HELDOUT_TWO = [
    (["ur-minarepakistan", "fr-viaducdemillau"], "comparison", "en",
     "Which was completed earlier, the Minar-e-Pakistan or the Millau Viaduct? Give the completion year of each.",
     "The Minar-e-Pakistan (completed 1968) was completed before the Millau Viaduct (opened 2004).",
     [["1968", "۱۹۶۸"], ["2004"]]),
    (["ta-kallanai", "te-golconda"], "multi_hop", "ta",
     "கல்லணையின் உயரம் எத்தனை அடி? கோல்கொண்டா கோட்டை எத்தனை மீட்டர் உயரமுள்ள மலையின் மீது கட்டப்பட்டது?",
     "கல்லணையின் உயரம் 18 அடி; கோல்கொண்டா கோட்டை 120 மீட்டர் உயரமுள்ள மலையின் மீது கட்டப்பட்டது.",
     [["18", "௧௮"], ["120", "౧౨౦", "௧௨௦"]]),
]


def load_pages(base: Path) -> dict[str, list[str]]:
    docs = json.loads((base / "documents.json").read_text(encoding="utf-8"))
    out = {}
    for d in docs:
        if "derived_from" in d:
            continue
        doc = pymupdf.open(base / "pdfs" / f"{d['doc_id']}.pdf")
        out[d["doc_id"]] = [normalize_text(p.get_text()) for p in doc]
    return out


def _ns(s: str) -> str:
    return "".join(s.split())


def fact_pages(pages: list[str], fact: list[str]) -> list[int]:
    hit = [i + 1 for i, t in enumerate(pages) if any(alias_matches(t, a, normalized=True) for a in fact)]
    if hit:
        return hit
    # second pass: the text layer sometimes inserts whitespace inside a word (CJK digits, i-matra)
    hit = [i + 1 for i, t in enumerate(pages) if any(_ns(normalize_text(a)) in _ns(t) for a in fact)]
    if hit:
        print("WARN whitespace-only match:", fact[:3])
    return hit


def evidence_for(pages_by_doc, docs, facts, qid):
    """Smallest page set per doc covering the facts (each fact on >=1 listed doc)."""
    ev: dict[str, set[int]] = {d: set() for d in docs}
    for f in facts:
        hits = [(d, fact_pages(pages_by_doc[d], f)) for d in docs]
        hits = [(d, p) for d, p in hits if p]
        if not hits:
            raise SystemExit(f"{qid}: fact {f[:3]} not found in {docs}")
        for d, p in hits:
            if not ev[d] & set(p):
                ev[d].add(p[0])
    for d in docs:
        if not ev[d]:
            raise SystemExit(f"{qid}: doc {d} contributes no fact")
    return [{"doc_id": d, "pages": sorted(ev[d])} for d in docs]


def snippet(base: Path, doc_id: str, facts) -> str:
    src = (base / "sources" / f"{doc_id}.txt").read_text(encoding="utf-8")
    low = src.lower()
    parts = []
    for f in facts:
        for a in f:
            i = low.find(a.lower())
            if i >= 0:
                s = src[max(0, i - 70): i + len(a) + 50].replace("\n", " ").strip()
                parts.append(f'"...{s}..."')
                break
    return " | ".join(parts)


def rec(qid, doc_id, cat, q, ans, facts, ev_pages, notes, ql, al, doc_ids, evidence, combo):
    return {"q_id": qid, "doc_id": doc_id, "category": cat, "question": q,
            "answerable": cat != "unanswerable", "reference_answer": ans, "required_facts": facts,
            "evidence_pages": ev_pages, "notes": notes, "question_language": ql, "answer_language": al,
            "doc_ids": doc_ids, "evidence": evidence, "combo": combo}


def single(base, pages, prefix, spec, combo=None, doc_override=None, q_override=None, doc_ids=None, scan=False):
    doc, c, cat, ql, al, q, ans, facts = spec
    combo = combo or (f"en-on-{doc.split('-')[0]}" if c == "en" else c)
    tag = {"same": "same", "cross": "cross", "unanswerable": "unans", "answer-in": "answerin",
           "library": "lib", "scanned": "scan"}.get(combo, "en")
    qid = f"{prefix}-{doc}-{tag}"
    target = doc_override or doc
    if cat == "unanswerable":
        return rec(qid, target, cat, q_override or q, ans, [], [], "Not stated anywhere in the document (checked by reading the whole source text).",
                   ql, al, doc_ids or [target], [], combo)
    ev = evidence_for(pages, [doc], facts, qid)
    p = ev[0]["pages"]
    note = snippet(base, doc, facts)
    if scan:
        note = f"Scanned copy of {doc}; page numbers are pages of the scanned PDF (same as the source PDF). " + note
    return rec(qid, target, cat, q_override or q, ans, facts, p, note, ql, al, doc_ids or [target],
               [{"doc_id": target, "pages": p}], combo)


def build(base: Path, prefix: str, specs, two, library, scanned: bool) -> list[dict]:
    pages = load_pages(base)
    out = []
    by_doc_same = {s[0]: s for s in specs if s[1] == "same"}
    for s in specs:
        out.append(single(base, pages, prefix, s))
    for doc, q in library.items():
        out.append(single(base, pages, prefix, by_doc_same[doc], combo="library", q_override=q, doc_ids="all"))
    for i, (docs, cat, ql, q, ans, facts) in enumerate(two, start=1):
        qid = f"{prefix}-twobooks-{i:02d}"
        ev = evidence_for(pages, docs, facts, qid)
        note = " | ".join(f"{d}: {snippet(base, d, facts)}" for d in docs)
        out.append(rec(qid, None, cat, q, ans, facts, None, note, ql, ql, "all", ev, "two-books"))
    if scanned:
        docs = json.loads((base / "documents.json").read_text(encoding="utf-8"))
        for d in docs:
            if "derived_from" not in d:
                continue
            src = d["derived_from"]
            if src not in by_doc_same:
                continue
            out.append(single(base, pages, prefix, by_doc_same[src], combo="scanned", doc_override=d["doc_id"],
                              doc_ids=[d["doc_id"]], scan=True))
    ids = [r["q_id"] for r in out]
    assert len(ids) == len(set(ids)), "duplicate q_id"
    return out


def write(path: Path, recs):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    print(path, len(recs))


if __name__ == "__main__":
    B = ROOT / "benchmark" / "languages"
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if which == "dev":
        write(B / "questions.jsonl", build(B, "lg", DEV, TWO_BOOKS, LIBRARY, scanned=True))
    else:
        write(B / "heldout" / "questions.jsonl", build(B / "heldout", "lgh", HELDOUT, HELDOUT_TWO, {}, scanned=False))
