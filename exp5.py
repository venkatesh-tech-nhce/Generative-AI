!pip install -q crewai

import os
from getpass import getpass
from crewai import Agent, Task, Crew, LLM

# Exp 5: Multi-agent travel planner
class TravelPlanner:
    def __init__(self):
        os.environ["GEMINI_API_KEY"] = getpass(
            "Enter Gemini API Key: "
        )

        self.llm = LLM(
            model="gemini/gemini-3.6-flash",
            api_key=os.environ["GEMINI_API_KEY"]
        )

    def create_agent(self, role, goal, backstory):
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=self.llm
        )

    def validate(self, days, people, budget):
        if days <= 0:
            raise ValueError("Number of days must be greater than 0.")

        if people <= 0:
            raise ValueError("Number of travelers must be greater than 0.")

        if budget <= 0:
            raise ValueError("Budget must be greater than 0.")

    def run(self):
        # Original input code retained
        source = input("Starting city: ")
        destination = input("Destination: ")
        days = input("Number of days: ")
        people = input("Number of travelers: ")
        budget = input("Budget (₹): ")
        interest = input("Interests: ")

        # Old:
        # info = f"""
        # From: {source}
        # To: {destination}
        # Days: {days}
        # Travelers: {people}
        # Budget: {budget} ₹
        # Interests: {interest}
        # """

        days = int(days)
        people = int(people)
        budget = float(budget)

        self.validate(days, people, budget)

        info = f"""
From: {source}
To: {destination}
Days: {days}
Travelers: {people}
Budget: ₹{budget}
Interests: {interest}
"""

        transport = self.create_agent(
            "Transport Expert",
            "Suggest affordable transportation",
            "You are a travel transport expert."
        )

        hotel = self.create_agent(
            "Hotel Expert",
            "Suggest affordable accommodation",
            "You are a hotel expert."
        )

        activities = self.create_agent(
            "Activity Expert",
            "Suggest activities",
            "You are a local travel guide."
        )

        planner = self.create_agent(
            "Travel Planner",
            "Create a complete budget-friendly itinerary",
            "You are an experienced travel planner."
        )

        t1 = Task(
            description=f"""
Suggest affordable transport for:
{info}

Check that the transportation is suitable for the number
of travelers and trip duration.
""",
            agent=transport,
            expected_output="Transport recommendation and cost."
        )

        t2 = Task(
            description=f"""
Suggest affordable hotels for:
{info}

Check room requirements for the number of travelers.
""",
            agent=hotel,
            expected_output="Hotel recommendation and cost."
        )

        t3 = Task(
            description=f"""
Suggest realistic activities for:
{info}

Activities must match the destination and interests.
""",
            agent=activities,
            expected_output="Activities and estimated costs."
        )

        # Old:
        # t4 = Task(
        #     description=f"""
        #     Create a day-wise itinerary using the recommendations
        #     from the other agents.
        #     Requirements:
        #     {info}
        #     Keep the total cost within the budget.
        #     """,
        #     agent=planner,
        #     expected_output="Complete travel itinerary."
        # )

        t4 = Task(
            description=f"""
Create a day-wise itinerary using the recommendations
from the other agents.

Requirements:
{info}

Validate:
1. Total estimated cost must stay within the budget.
2. Activities must match the destination.
3. Activities must match the user's interests.
4. Schedule must fit within the number of days.
5. Transport must match the route.
6. Avoid duplicate activities.

If information is uncertain, clearly mention it.
""",
            agent=planner,
            expected_output="Validated complete travel itinerary."
        )

        crew = Crew(
            agents=[
                transport,
                hotel,
                activities,
                planner
            ],
            tasks=[
                t1,
                t2,
                t3,
                t4
            ]
        )

        result = crew.kickoff()

        print("\n===== FINAL TRAVEL ITINERARY =====")
        print(result)


TravelPlanner().run()