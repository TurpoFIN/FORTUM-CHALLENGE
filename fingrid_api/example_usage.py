import datetime as dt

from . import (
    FingridApiClient,
    get_fi_ee_transfer,
    get_fi_no4_transfer,
    get_fi_se1_transfer,
    get_fi_se3_transfer,
)


def main() -> None:
    client = FingridApiClient()
    end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    start = end - dt.timedelta(hours=72)

    df_ee = get_fi_ee_transfer(client, start, end)
    df_no4 = get_fi_no4_transfer(client, start, end)
    df_se1 = get_fi_se1_transfer(client, start, end)
    df_se3 = get_fi_se3_transfer(client, start, end)

    print("FI-EE head:\n", df_ee.head(), "\n")
    print("FI-NO4 head:\n", df_no4.head(), "\n")
    print("FI-SE1 head:\n", df_se1.head(), "\n")
    print("FI-SE3 head:\n", df_se3.head(), "\n")


if __name__ == "__main__":
    main()


