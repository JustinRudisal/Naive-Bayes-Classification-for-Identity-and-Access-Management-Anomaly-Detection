"""Module to simulate and log single sign-on (SSO) events for security monitoring."""

from datetime import datetime, timedelta
import random
import pandas as pd

# Setup basic parameters for the simulation
NUM_RECORDS = 10000  # Total number of records to generate
START_DATE = datetime(2024, 1, 1)  # Start date for the simulation
END_DATE = datetime(2024, 4, 1)  # End date for the simulation
USERS = [str(i) for i in range(1, 101)]  # List of user IDs
APPLICATIONS = ["App" + str(i) for i in range(1, 21)]  # List of application IDs

# Setup IP addresses for the simulation
PRIMARY_IPS = {
    user: f"192.168.{i % 255}.{random.randint(1, 254)}" for i, user in enumerate(USERS)
}  # Primary IP addresses for each user
SECONDARY_IPS = [
    f"10.10.{i}.{random.randint(1, 254)}" for i in range(100)
]  # List of secondary IP addresses
GLOBAL_IPS = [
    f"172.16.{i}.{random.randint(1, 254)}" for i in range(100)
]  # List of global IP addresses

RECORDS = []  # List to store the generated records


# Function to simulate a single SSO event
# pylint: disable=too-many-arguments
def simulate_activity(
    activity_timestamp,
    activity_user_id,
    activity_application_id,
    activity_event_type,
    activity_ip_address,
    activity_outcome="Success",
    activity_reason=None,
    activity_is_malicious="No",
):
    """Simulate and return a single SSO event as a list."""
    return [
        activity_timestamp,
        activity_user_id,
        activity_application_id,
        activity_event_type,
        activity_outcome,
        activity_reason,
        activity_ip_address,
        activity_is_malicious,
    ]


# Generate general and malicious activities
for _ in range(NUM_RECORDS):
    # Generate a random timestamp within the simulation period
    timestamp = START_DATE + timedelta(
        seconds=random.randint(0, round((END_DATE - START_DATE).total_seconds()))
    )
    user_id = random.choice(USERS)  # Select a random user
    application_id = random.choice(APPLICATIONS)  # Select a random application
    # Select an IP address, with a 90% chance of selecting the user's primary IP
    ip_address = (
        PRIMARY_IPS[user_id] if random.random() > 0.1 else random.choice(SECONDARY_IPS)
    )
    event_type = random.choice(
        ["LoginAttempt", "AccessRequest"]
    )  # Select a random event type
    # Determine the outcome and reason, with a 15% chance of failure
    outcome, reason = (
        ("Success", None)
        if random.random() > 0.15
        else ("Failure", "InvalidCredentials")
    )
    # Determine if the event is malicious
    IS_MALICIOUS = (
        "Yes" if ip_address in SECONDARY_IPS and outcome == "Failure" else "No"
    )
    # Generate the event and add it to the list of records
    record = simulate_activity(
        timestamp,
        user_id,
        application_id,
        event_type,
        ip_address,
        outcome,
        reason,
        IS_MALICIOUS,
    )
    RECORDS.append(record)

# Additional activity simulation for geolocation and repeated failures
for user_id in random.sample(USERS, 15):
    for _ in range(random.randint(2, 5)):
        application_id = random.choice(APPLICATIONS)
        timestamps = [
            START_DATE
            + timedelta(
                seconds=random.randint(
                    0, round((END_DATE - START_DATE).total_seconds())
                )
            )
            for _ in range(2)
        ]
        for timestamp in sorted(timestamps):
            ip_address = random.choice(GLOBAL_IPS)
            record = simulate_activity(
                timestamp,
                user_id,
                application_id,
                "LoginAttempt",
                ip_address,
                "Success",
                None,
                "Yes",
            )
            RECORDS.append(record)

for user_id in random.sample(USERS, 20):
    application_id = random.choice(APPLICATIONS)
    timestamp = START_DATE + timedelta(
        seconds=random.randint(0, round((END_DATE - START_DATE).total_seconds()))
    )
    for _ in range(random.randint(3, 6)):
        ip_address = random.choice(SECONDARY_IPS)
        record = simulate_activity(
            timestamp,
            user_id,
            application_id,
            "AccessRequest",
            ip_address,
            "Failure",
            "NoAccessRights",
            "Yes",
        )
        RECORDS.append(record)

# Convert the list of records to a DataFrame, sort it by timestamp, and save it as a CSV file
df = pd.DataFrame(
    RECORDS,
    columns=[
        "Timestamp",
        "UserID",
        "ApplicationID",
        "EventType",
        "Outcome",
        "Reason",
        "IPAddress",
        "IsMalicious",
    ],
)
df.sort_values("Timestamp", inplace=True)
df.to_csv("sso_events.csv", index=False)

print(df.head())
