#Hashim Waqar
#Cp493
#gradio_app.py


import gradio as gr
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000"
STEP_GOAL = 10000  # daily target steps


def visualize_steps(user_id: str):
    try:
        res = requests.get(f"{API_URL}/steps/{user_id}")
        if res.status_code != 200:
            return pd.DataFrame(), plt.figure()

        data = res.json().get("steps", [])
        if not data:
            return pd.DataFrame(), plt.figure()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])

        fig, ax = plt.subplots()
        ax.plot(df["date"], df["steps"], marker="o", linestyle="-", label="Actual Steps")

        # Try fetching prediction for visualization
        try:
            pred_res = requests.get(f"{API_URL}/predict/{user_id}")
            if pred_res.status_code == 200:
                prediction = pred_res.json().get("prediction", "")
                if "Predicted" in prediction:
                    digits = [c for c in prediction if c.isdigit()]
                    if digits:
                        pred_value = int("".join(digits))
                        next_day = df["date"].max() + pd.Timedelta(days=1)
                        ax.plot(
                            [df["date"].max(), next_day],
                            [df["steps"].iloc[-1], pred_value],
                            linestyle="--",
                            label="Predicted (Tomorrow)",
                        )
        except Exception as e:
            print("Prediction fetch error:", e)

        ax.set_xlabel("Date")
        ax.set_ylabel("Steps")
        ax.set_title(f"Daily Steps for {user_id}")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        return df, fig

    except Exception as e:
        print("Error:", e)
        return pd.DataFrame(), plt.figure()


def get_insights(user_id: str):
    try:
        res = requests.get(f"{API_URL}/insights/{user_id}")
        if res.status_code == 200:
            insights = res.json().get("insights", [])
            return "\n".join(insights) if insights else "No insights found."
        return "Unable to get insights."
    except Exception as e:
        return f"Error: {e}"


def get_prediction(user_id: str):
    try:
        res = requests.get(f"{API_URL}/predict/{user_id}")
        if res.status_code == 200:
            return res.json().get("prediction", "No prediction found.")
        return "Unable to get prediction."
    except Exception as e:
        return f"Error: {e}"


def get_prediction_history(user_id: str):
    try:
        res = requests.get(f"{API_URL}/predictions/{user_id}")
        if res.status_code == 200:
            preds = res.json().get("predictions", [])
            if preds:
                df = pd.DataFrame(preds)
                df["date"] = pd.to_datetime(df["date"])
                return df
        return pd.DataFrame(columns=["date", "prediction"])
    except Exception as e:
        print("Error fetching prediction history:", e)
        return pd.DataFrame(columns=["date", "prediction"])


def daily_summary(user_id: str):
    """Combine latest steps, prediction and AI insights into one text block."""
    # Get prediction text
    prediction_text = get_prediction(user_id)

    # Get AI insights text
    insights_text = get_insights(user_id)

    # Latest recorded steps
    try:
        res = requests.get(f"{API_URL}/steps/{user_id}")
        if res.status_code == 200:
            steps_data = res.json().get("steps", [])
        else:
            steps_data = []
    except Exception:
        steps_data = []

    if steps_data:
        df = pd.DataFrame(steps_data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        latest = df.iloc[-1]
        latest_date = latest["date"].strftime("%b %d, %Y")
        latest_line = (
            f"Latest recorded day: {latest_date} — "
            f"{int(latest['steps']):,} steps.\n\n"
        )
    else:
        latest_line = "No step data found yet for this user.\n\n"

    goal_line = f"Daily goal: {STEP_GOAL:,} steps.\n\n"

    summary = (
        f"User: {user_id}\n\n"
        f"{goal_line}"
        f"{latest_line}"
        f"Tomorrow's prediction:\n{prediction_text}\n\n"
        f"Coach's insights:\n{insights_text}"
    )
    return summary



def chat_step(user_id: str, message: str, history):
    if not user_id:
        reply = "Please enter your user ID above before chatting."
        history = history + [[message, reply]]
        return "", history

    payload = {
        "user_id": user_id,
        "history": history,
        "message": message,
    }

    try:
        res = requests.post(f"{API_URL}/chat", json=payload)
        if res.status_code == 200:
            reply = res.json().get("reply", "No reply from AI.")
        else:
            reply = f"Error from server: {res.status_code}"
    except Exception as e:
        reply = f"Error contacting AI: {e}"

    history = history + [[message, reply]]
  
    return "", history


# Gradio UI layout
with gr.Blocks(title="Fitbit Steps Dashboard") as demo:
    gr.Markdown("# Fitbit Steps Dashboard")
    gr.Markdown("Easily view your step data, daily insights, and predictions.")

    # Today Summary first so it's the default tab
    with gr.Tab("Today Summary"):
        user_input_summary = gr.Textbox(
            label="User ID", placeholder="Enter your user ID"
        )
        summary_output = gr.Textbox(label="Daily Summary", lines=12)
        gr.Button("Get Today Summary").click(
            daily_summary,
            inputs=user_input_summary,
            outputs=summary_output,
        )

    with gr.Tab("Visualization"):
        user_input_viz = gr.Textbox(
            label="Name or User ID", placeholder="Enter your user ID"
        )
        data_output = gr.Dataframe(label="Steps")
        plot_output = gr.Plot(label="Steps Over Time")
        gr.Button("Show Data").click(
            visualize_steps,
            inputs=user_input_viz,
            outputs=[data_output, plot_output],
        )

    with gr.Tab("Insights"):
        user_input_insight = gr.Textbox(
            label="Name or User ID", placeholder="Enter your user ID"
        )
        insight_output = gr.Textbox(label="Insights", lines=6)
        gr.Button("Generate Insights").click(
            get_insights,
            inputs=user_input_insight,
            outputs=insight_output,
        )

    with gr.Tab("Prediction"):
        user_input_pred = gr.Textbox(
            label="User ID", placeholder="Enter your Name or user ID"
        )
        prediction_output = gr.Textbox(label="Prediction", lines=2)
        gr.Button("Predict Tomorrow's Steps").click(
            get_prediction,
            inputs=user_input_pred,
            outputs=prediction_output,
        )

    with gr.Tab("Prediction History"):
        user_input_hist = gr.Textbox(
            label="User ID", placeholder="Enter your user ID"
        )
        history_output = gr.Dataframe(label="Prediction History")
        gr.Button("Show History").click(
            get_prediction_history,
            inputs=user_input_hist,
            outputs=history_output,
        )

    with gr.Tab("Chat with Coach"):
        chat_user_id = gr.Textbox(
            label="User ID", placeholder="Enter your user ID"
        )
        chatbot = gr.Chatbot(label="Conversation")
        chat_message = gr.Textbox(
            label="Your message",
            placeholder="Ask your coach anything about your steps, goals, or habits...",
            lines=2,
        )
        chat_send = gr.Button("Send")

        chat_send.click(
            chat_step,
            inputs=[chat_user_id, chat_message, chatbot],
            outputs=[chat_message, chatbot],
        )


if __name__ == "__main__":
    demo.launch()

