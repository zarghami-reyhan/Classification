import numpy as np


train_texts = [
    # نظرات مثبت (برچسب 1)
    ("غذای این رستوران فوق‌العاده خوشمزه و تازه بود. برخورد پرسنل عالی بود و حتما دوباره مراجعه می‌کنم.", 1),
    ("کیفیت پیتزا عالی بود، نان تازه و پنیر کش‌دار. محیط هم خیلی تمیز و شیک بود.", 1),
    ("استیک عالی پخته شده بود و سس مخصوص بی‌نظیر داشت. پیشنهاد می‌کنم حتما امتحان کنید.", 1),
    ("سفارش ما سریع آماده شد. کباب گرم و خوش‌طعم بود، رفتار گارسون‌ها هم کاملا محترمانه بود.", 1),
    ("دیزاین بشقاب‌ها عالی و سالاد بسیار تازه بود. یکی از بهترین تجربه های من بود.", 1),
    ("پاستا آلفردو فوق‌العاده طعم خوبی داشت. محیط آرام و موسیقی ملایم هم لذت غذا را چند برابر کرد.", 1),
    ("برگر باکیفیت و آبدار بود. سیب‌زمینی سرخ‌کرده هم ترد و تازه بود. ممنون از کیفیت خوبتون.", 1),
    ("قیمت‌ها نسبت به کیفیت عالی بود. سرویس‌دهی سریع و تمیز بود و حتما مشتری دائم می‌شم.", 1),
    ("سوپ روز عالی و گرم بود. تمام جزئیات رعایت شده بود و برخورد مدیریت بسیار عالی بود.", 1),
    ("چلوکباب زعفرانی عالی و گوشت تازه بود. پیک هم غذا را کاملا داغ تحویل داد.", 1),

    # نظرات منفی (برچسب 0)
    ("غذا بسیار سرد و بی‌کیفیت بود. گوشت کباب شور بود و اصلاً نتوانستیم بخوریم.", 0),
    ("معطلی زیادی داشتیم. بعد از یک ساعت غذای اشتباه و سرد آوردند و برخورد پرسنل هم بد بود.", 0),
    ("قیمت‌ها بسیار گران اما کیفیت غذا افتضاح بود. سالاد کهنه و پلاسیده بود.", 0),
    ("پیتزا کاملا سوخته بود و طعم بد می‌داد. وقتی اعتراض کردیم برخورد زشت و نامناسبی داشتند.", 0),
    ("محیط کثیف و شلوغ بود. روی میزها پاک نشده بود و سرویس‌دهی فوق‌العاده ضعیف و دیر بود.", 0),
    ("برگر کاملا خام و بی‌طعم بود. ارزش این قیمت گران را نداشت و اصلاً پیشنهاد نمی‌کنم.", 0),
    ("غذای ما با تأخیر دو ساعته رسید و کاملا یخ کرده بود. پیگیری پشتیبانی هم صفر بود.", 0),
    ("کیفیت نسبت به قبل خیلی افت کرده. مرغ بوی بد می‌داد و برنج هم کاملا خمیر بود.", 0),
    ("سفارش ناقص ارسال شد. نوشابه و سس را فراموش کرده بودند و پاسخگویی تلفن هم بد بود.", 0),
    ("ماهی بی‌کیفیت و شور بود. فضای رستوران بوی نا مطبوعی می‌داد و اصلا راضی نبودیم.", 0)
]

test_texts = [
    ("کباب عالی و داغ بود، برخورد پرسنل بسیار محترمانه بود و حتما باز هم میام.", 1),
    ("سوپ عالی و خوشمزه بود اما سالاد کمی کهنه بود ولی در کل خوب بود.", 1),
    ("غذا بسیار دیر رسید و کاملا سرد و بی‌کیفیت بود. برخورد پیک هم خیلی بد بود.", 0),
    ("قیمت بسیار گران و کیفیت پیتزا افتضاح بود. اصلا پیشنهاد نمی‌کنم.", 0)
]



pos_words = ["عالی", "خوشمزه", "تازه", "فوق‌العاده", "تمیز", "خوب", "گرم", "بی‌نظیر", "باکیفیت", "ترد", "محترمانه"]
neg_words = ["بد", "سرد", "بی‌کیفیت", "افتضاح", "کثیف", "شور", "گران", "سوخته", "خام", "بوی نا", "خمیر", "کهنه"]
amp_words = ["حتما", "پیشنهاد", "بهترین", "دوباره", "دائم", "مجدد"]
crit_words = ["معطلی", "دیر", "زشت", "تأخیر", "پیگیری", "صفر", "پاسخگویی", "اعتراض", "فراموش"]

def extract_features(text):
    words = text.split()
    
    f1 = sum(1 for w in words if any(pw in w for pw in pos_words))
    f2 = sum(1 for w in words if any(nw in w for nw in neg_words))
    f3 = 1.0 if any(aw in text for aw in amp_words) else 0.0
    f4 = sum(1 for w in words if any(cw in w for cw in crit_words))
    f5 = len(words)
    
    return np.array([f1, f2, f3, f4, f5], dtype=float)

X_train = np.array([extract_features(t[0]) for t in train_texts])
y_train = np.array([t[1] for t in train_texts])

X_test = np.array([extract_features(t[0]) for t in test_texts])
y_test = np.array([t[1] for t in test_texts])

mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8 

X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def compute_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

np.random.seed(42)
weights = np.random.randn(5) * 0.01
bias = 0.0

epochs = 100
learning_rate = 0.1

print("--- آغاز آموزش مدل پرسپترون (SGD) ---")
for epoch in range(epochs):
    indices = np.arange(len(X_train_norm))
    np.random.shuffle(indices)
    
    for idx in indices:
        x_i = X_train_norm[idx]
        y_i = y_train[idx]
        
        z = np.dot(x_i, weights) + bias
        y_hat = sigmoid(z)
        
        error = y_hat - y_i
        dw = error * x_i
        db = error
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
    if (epoch + 1) % 20 == 0 or epoch == 0:
        z_all = np.dot(X_train_norm, weights) + bias
        preds_all = sigmoid(z_all)
        loss = compute_loss(y_train, preds_all)
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss:.4f}")



print("\n--- Test---")
z_test = np.dot(X_test_norm, weights) + bias
probs_test = sigmoid(z_test)
preds_test = (probs_test >= 0.5).astype(int)

for i, (text, label) in enumerate(test_texts):
    pred_label = "مثبت" if preds_test[i] == 1 else "منفی"
    true_label = "مثبت" if label == 1 else "منفی"
    print(f"\nنظر {i+1}: {text}")
    print(f"  احتمال پیش‌بینی مثبت: {probs_test[i]:.4f}")
    print(f"  برچسب واقعی: {true_label} | برچسب پیش‌بینی شده: {pred_label}")

# ==========================================
# ۵. محاسبه معیارهای Accuracy, Precision, Recall
# ==========================================

TP = np.sum((preds_test == 1) & (y_test == 1))
TN = np.sum((preds_test == 0) & (y_test == 0))
FP = np.sum((preds_test == 1) & (y_test == 0))
FN = np.sum((preds_test == 0) & (y_test == 1))

accuracy = (TP + TN) / len(y_test) if len(y_test) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

print("\n==========================================")
print("معیارهای ارزیابی مدل روی داده‌های آزمایشی:")
print("==========================================")
print(f"دقت کلی (Accuracy)  : {accuracy * 100:.2f}%")
print(f"معیار Precision     : {precision * 100:.2f}%")
print(f"معیار Recall        : {recall * 100:.2f}%")
