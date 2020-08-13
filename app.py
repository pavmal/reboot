import os, time
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import Length
import models

SECRET_KEY = os.urandom(32)

print("Preparing classifier")
start_time = time.time()
ipoteka_review_classifier = models.IpotekaReviewsToneSentimentClassifier()
print("Classifiers are ready")
print(time.time() - start_time, "seconds")


class DemoModel(FlaskForm):
    """
    ipoteka_tone: поле формы для ввода текста для анализа
    submit: кнопка отправки текста на обработку
    """
    ipoteka_tone = TextAreaField('Текст для анализа тональности', validators=[Length(min=0, max=250)])
    submit = SubmitField('Отправить на обработку')


app = Flask(__name__)
app.secret_key = SECRET_KEY


@app.route('/', methods=['GET', 'POST'])
def render_main():
    form = DemoModel()
    if request.method == "POST":
        # for reviews analysis
        user_text_ipoteka = form.ipoteka_tone.data
        if user_text_ipoteka != '':
            predictions_ipoteka = ipoteka_review_classifier.get_prediction_message(user_text_ipoteka).split(',')
            if predictions_ipoteka[1] == 'positive':  # class 1 - positive feedback
                class_res_ipoteka = 'POSITIVE FEEDBACK'
                persent_ipoteka = predictions_ipoteka[2]
            elif predictions_ipoteka[1] == 'negative':
                class_res_ipoteka = 'NEGATIVE FEEDBACK'
                persent_ipoteka = str(100 - float(predictions_ipoteka[2]))
            else:
                class_res_ipoteka = 'NEUTRAL'
                persent_ipoteka = predictions_ipoteka[2]
            prediction_message_ipoteka = 'Ваш отзыв классифицирован как: {} c вероятностью: {} %'.format(
                class_res_ipoteka,
                persent_ipoteka)
        else:
            prediction_message_ipoteka = None

        return render_template('demo.html', form=form, result_ipoteka=prediction_message_ipoteka)
    else:
        return render_template('demo.html', form=form, result_ipoteka=None)


@app.route('/about/')
def render_about():
    """
    Представление страницы "О сервисе"
    :return: Забавная картинка
    """
    return render_template('about.html')


if __name__ == '__main__':
    #app.run('127.0.0.1', 6001, debug=True)
    app.run()  # for gunicorn server
