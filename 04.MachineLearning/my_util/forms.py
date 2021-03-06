from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, Email, EqualTo


class RegistrationForm(FlaskForm):
    username = StringField("문의내용",
                           validators=[DataRequired(), Length(min=4, max=50)])
    email = StringField("답변받을 e-mail",
                        validators=[DataRequired(), Email()])
    password = PasswordField("비밀번호",
                             validators=[DataRequired(), Length(min=4, max=20)])
    confirm_password = PasswordField("비밀번호 확인",
                                     validators=[DataRequired(), EqualTo("password")])
    submit = SubmitField("문의")
