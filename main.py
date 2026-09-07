import argparse
import math
import random
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_SIZE = 1000
MIN_VALUE = -10000
MAX_VALUE = 10000
RANDOM_SEED = 42
RANGE_STEP = 1000

SIGN_ORDER = ("Отрицательное", "Ноль", "Положительное")
SIGN_COLORS = {
    "Отрицательное": "#c0392b",
    "Ноль": "#7f8c8d",
    "Положительное": "#1e8449",
}


# Все файлы создаются рядом с main.py, как описано в отчете.
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_FILE = PROJECT_DIR / "dataset.csv"
DATAFRAME_FILE = PROJECT_DIR / "dataframe.csv"
RANGES_FILE = PROJECT_DIR / "ranges.csv"
COMBINED_FILE = PROJECT_DIR / "combined_dataframe.csv"
RESULT_FILE = PROJECT_DIR / "result.txt"
LINEAR_PLOT_FILE = PROJECT_DIR / "linear_plot.png"
HISTOGRAM_FILE = PROJECT_DIR / "histogram.png"
SORTED_PLOT_FILE = PROJECT_DIR / "sorted_values.png"
COMBINED_PLOT_FILE = PROJECT_DIR / "combined_dependence.png"


def generate_dataset() -> pd.Series:
    """Сгенерировать 1000 целых чисел в диапазоне [-10000; 10000]."""
    random.seed(RANDOM_SEED)
    data = [random.randint(MIN_VALUE, MAX_VALUE) for _ in range(DATASET_SIZE)]
    return pd.Series(data, name="Значение")


def print_dataset_info(series: pd.Series) -> None:
    """Вывести сводную информацию об исходном наборе данных."""
    print("Набор данных сформирован")
    print("Количество элементов:", series.count())
    print("Тип данных:", series.dtype)
    print("Первые 10 значений:")
    print(series.head(10).to_string())

    missing_values = int(series.isna().sum())
    incorrect_values = series[(series < MIN_VALUE) | (series > MAX_VALUE)]
    duplicated_values = int(series.duplicated().sum())

    print("\nПредварительная проверка данных")
    print("Количество пропущенных значений:", missing_values)
    print("Количество значений вне диапазона:", int(incorrect_values.count()))
    print("Количество повторяющихся элементов:", duplicated_values)


def clean_data(series: pd.Series) -> pd.Series:
    """Очистить Series от пропусков, нечисловых и выходящих за диапазон значений."""
    cleaned = pd.to_numeric(series, errors="coerce")
    cleaned = cleaned.dropna()
    cleaned = cleaned[(cleaned >= MIN_VALUE) & (cleaned <= MAX_VALUE)]

    # В отчете отдельно указана проверка целочисленности.
    # После генерации все элементы уже целые, но при загрузке измененного CSV
    # нецелые значения будут исключены.
    integer_mask = np.isclose(cleaned.to_numpy(dtype=float), np.round(cleaned.to_numpy(dtype=float)))
    cleaned = cleaned[integer_mask]
    cleaned = cleaned.astype(int).reset_index(drop=True)
    cleaned.name = "Значение"
    return cleaned


def get_range_bins_and_labels() -> tuple[np.ndarray, list[str]]:
    """Границы и подписи диапазонов шириной 1000 на отрезке [-10000; 10000]."""
    edges = np.arange(MIN_VALUE, MAX_VALUE + RANGE_STEP, RANGE_STEP, dtype=int)
    cut_edges = edges.copy()
    # Последнюю правую границу расширяем на 1, чтобы значение 10000 попало в интервал.
    cut_edges[-1] = MAX_VALUE + 1

    labels: list[str] = []
    for i in range(len(edges) - 1):
        start = int(edges[i])
        if i == len(edges) - 2:
            end = MAX_VALUE
        else:
            end = int(edges[i + 1] - 1)
        labels.append(f"{start}..{end}")
    return cut_edges, labels


def classify_by_sign(series: pd.Series) -> pd.Series:
    """Разделить значения по знаку: отрицательное / ноль / положительное."""
    values = series.to_numpy()
    labels = np.where(values < 0, "Отрицательное", np.where(values > 0, "Положительное", "Ноль"))
    return pd.Series(
        pd.Categorical(labels, categories=list(SIGN_ORDER), ordered=True),
        index=series.index,
        name="Знак",
    )


def classify_by_range(series: pd.Series) -> pd.Series:
    """Разделить значения по диапазонам шириной 1000."""
    cut_edges, labels = get_range_bins_and_labels()
    groups = pd.cut(
        series,
        bins=cut_edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    return pd.Series(groups, index=series.index, name="Диапазон")


def calculate_statistics(series: pd.Series) -> dict[str, float | int]:
    """Рассчитать характеристики, перечисленные в отчете."""
    mean_value = series.mean()
    std_value = math.sqrt(
        sum((value - mean_value) ** 2 for value in series) / len(series)
    )

    return {
        "Минимальное значение": int(series.min()),
        "Максимальное значение": int(series.max()),
        "Сумма чисел": int(series.sum()),
        "Среднее значение": round(float(mean_value), 2),
        "Количество повторяющихся элементов": int(series.duplicated().sum()),
        "Количество уникальных значений с повторами": int((series.value_counts() > 1).sum()),
        "Среднеквадратическое отклонение": round(std_value, 2),
    }


def build_dataframe(series: pd.Series) -> pd.DataFrame:
    """
    Сформировать DataFrame из Series.

    Помимо исходных и отсортированных значений явно хранятся признаки
    разделения: знак числа и диапазон.
    """
    reset = series.reset_index(drop=True)
    sorted_ascending = reset.sort_values(ascending=True).reset_index(drop=True)
    sorted_descending = reset.sort_values(ascending=False).reset_index(drop=True)

    return pd.DataFrame(
        {
            "Исходные значения": reset,
            "Знак": classify_by_sign(reset),
            "Диапазон": classify_by_range(reset),
            "По возрастанию": sorted_ascending,
            "По убыванию": sorted_descending,
        }
    )


def format_dataframe_summary(dataframe: pd.DataFrame) -> str:
    """Сводная информация о DataFrame: размер, типы, пропуски, info(), describe()."""
    rows, cols = dataframe.shape
    buffer = StringIO()
    dataframe.info(buf=buffer)
    info_text = buffer.getvalue().strip()

    numeric_describe = dataframe.describe(include=[np.number]).to_string()
    missing = dataframe.isna().sum()

    lines = [
        f"Размер DataFrame: {rows} строк × {cols} столбцов",
        "",
        "Типы данных:",
        dataframe.dtypes.to_string(),
        "",
        f"Количество записей: {rows}",
        "",
        "Количество пропусков:",
        missing.to_string(),
        "",
        "info():",
        info_text,
        "",
        "describe():",
        numeric_describe,
    ]
    return "\n".join(lines)


def format_classification_summary(dataframe: pd.DataFrame) -> str:
    """Явное разделение данных по знаку и по диапазону."""
    sign_counts = (
        dataframe["Знак"]
        .value_counts(sort=False)
        .reindex(list(SIGN_ORDER), fill_value=0)
    )
    range_counts = dataframe["Диапазон"].value_counts(sort=False)

    lines = [
        "По знаку:",
        *[f"  {name}: {int(sign_counts[name])}" for name in SIGN_ORDER],
        "",
        "По диапазону:",
        *[f"  {label}: {int(count)}" for label, count in range_counts.items()],
    ]
    return "\n".join(lines)


def build_ranges_summary(series: pd.Series) -> pd.DataFrame:
    """
    Рассчитать суммарные показатели по диапазонам.

    Для каждого диапазона: количество, сумма, среднее, минимум, максимум.
    """
    temp = pd.DataFrame(
        {
            "Значение": series.reset_index(drop=True),
            "Диапазон": classify_by_range(series.reset_index(drop=True)),
        }
    )
    summary = (
        temp.groupby("Диапазон", sort=True, observed=True)["Значение"]
        .agg(Количество="count", Сумма="sum", Среднее="mean", Минимум="min", Максимум="max")
        .reset_index()
    )
    summary["Диапазон"] = summary["Диапазон"].astype(str)
    summary["Среднее"] = summary["Среднее"].round(2)
    return summary


def build_combined_dataframe(series: pd.Series) -> pd.DataFrame:
    """Объединённая структура показателей для выявления зависимостей."""
    reset = series.reset_index(drop=True)
    rounded = ((reset / 100).round() * 100).astype(int)
    return pd.DataFrame(
        {
            "Исходное значение": reset,
            "Округленное до сотен": rounded,
            "Абсолютное значение": reset.abs(),
            "Знак": classify_by_sign(reset),
            "Диапазон": classify_by_range(reset),
        }
    )


def save_text_report(
    statistics: dict[str, float | int],
    dataframe_summary: str,
    classification_summary: str,
    ranges: pd.DataFrame,
) -> None:
    """Вывести результаты в консоль и сохранить их в result.txt."""
    statistic_lines = [f"{name}: {value}" for name, value in statistics.items()]
    ranges_text = ranges.to_string(index=False)

    sections = [
        ("Стандартные числовые характеристики", "\n".join(statistic_lines)),
        ("Сводная информация о DataFrame", dataframe_summary),
        ("Разделение данных по указанным признакам", classification_summary),
        ("Суммарные данные по диапазонам", ranges_text),
    ]

    with RESULT_FILE.open("w", encoding="utf-8") as file:
        file.write("SimpleAnalysis — результаты анализа\n")
        for title, body in sections:
            file.write("\n")
            file.write(title + "\n")
            file.write("-" * len(title) + "\n")
            file.write(body.rstrip() + "\n")

    print("\nСтандартные числовые характеристики")
    print("\n".join(statistic_lines))
    print("\nСводная информация о DataFrame")
    print(dataframe_summary)
    print("\nРазделение данных по указанным признакам")
    print(classification_summary)
    print("\nСуммарные данные по диапазонам")
    print(ranges_text)


def plot_linear(series: pd.Series, show: bool) -> None:
    """Построить линейный график исходного набора данных."""
    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series.values)
    plt.title("Линейный график исходного набора данных")
    plt.xlabel("Порядковый номер")
    plt.ylabel("Значение")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LINEAR_PLOT_FILE, dpi=180)
    if show:
        plt.show()
    plt.close()


def plot_histogram(series: pd.Series, show: bool) -> None:
    """Округлить значения до сотен и построить гистограмму."""
    rounded_to_hundreds = (series / 100).round() * 100
    plt.figure(figsize=(10, 5))
    plt.hist(rounded_to_hundreds, bins=30, edgecolor="black")
    plt.title("Гистограмма значений, округленных до сотен")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(HISTOGRAM_FILE, dpi=180)
    if show:
        plt.show()
    plt.close()


def plot_sorted(dataframe: pd.DataFrame, show: bool) -> None:
    """Построить общий график значений по возрастанию и убыванию."""
    plt.figure(figsize=(10, 5))
    plt.plot(dataframe.index, dataframe["По возрастанию"], label="По возрастанию")
    plt.plot(dataframe.index, dataframe["По убыванию"], label="По убыванию")
    plt.title("Сравнение отсортированных значений")
    plt.xlabel("Порядковый номер")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SORTED_PLOT_FILE, dpi=180)
    if show:
        plt.show()
    plt.close()


def _scatter_by_sign(ax, combined: pd.DataFrame, x_col: str, y_col: str) -> None:
    for sign_name in SIGN_ORDER:
        part = combined[combined["Знак"] == sign_name]
        ax.scatter(
            part[x_col],
            part[y_col],
            s=14,
            alpha=0.45,
            color=SIGN_COLORS[sign_name],
            label=sign_name,
        )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True)


def plot_combined_dependence(combined: pd.DataFrame, show: bool) -> None:
    """
    Визуализировать объединённые показатели, чтобы показать зависимости:

    - округлённое до сотен почти совпадает с исходным значением;
    - абсолютное значение даёт V-образную зависимость от исходного;
    - знак и диапазон разделяют выборку на категории.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    _scatter_by_sign(ax, combined, "Исходное значение", "Округленное до сотен")
    limits = [MIN_VALUE, MAX_VALUE]
    ax.plot(limits, limits, color="black", linewidth=1, linestyle="--", label="y = x")
    ax.set_title("Зависимость: исходное → округлённое до сотен")
    ax.set_xlabel("Исходное значение")
    ax.set_ylabel("Округленное до сотен")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[0, 1]
    _scatter_by_sign(ax, combined, "Исходное значение", "Абсолютное значение")
    ax.set_title("Зависимость: исходное → абсолютное значение")
    ax.set_xlabel("Исходное значение")
    ax.set_ylabel("Абсолютное значение")

    ax = axes[1, 0]
    sign_counts = (
        combined["Знак"].value_counts(sort=False).reindex(list(SIGN_ORDER), fill_value=0)
    )
    bars = ax.bar(
        list(SIGN_ORDER),
        sign_counts.to_numpy(),
        color=[SIGN_COLORS[name] for name in SIGN_ORDER],
        edgecolor="black",
    )
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(int(sign_counts.max()) * 1.15, 1))
    ax.set_title("Разделение объединённых данных по знаку")
    ax.set_xlabel("Знак")
    ax.set_ylabel("Количество")
    ax.grid(True, axis="y")

    ax = axes[1, 1]
    grouped = combined.groupby("Диапазон", observed=True, sort=True).agg(
        Исходное=("Исходное значение", "mean"),
        Округленное=("Округленное до сотен", "mean"),
        Абсолютное=("Абсолютное значение", "mean"),
    )
    x_labels = grouped.index.astype(str)
    x_pos = np.arange(len(grouped))
    ax.plot(x_pos, grouped["Исходное"], marker="o", label="Среднее исходное")
    ax.plot(x_pos, grouped["Округленное"], marker="s", label="Среднее округлённое")
    ax.plot(x_pos, grouped["Абсолютное"], marker="^", label="Среднее абсолютное")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    ax.set_title("Зависимость средних показателей по диапазонам")
    ax.set_xlabel("Диапазон")
    ax.set_ylabel("Среднее значение")
    ax.legend(fontsize=8)
    ax.grid(True)

    fig.suptitle("Выявление зависимостей по объединённым показателям", fontsize=13)
    fig.tight_layout()
    fig.savefig(COMBINED_PLOT_FILE, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def validate_report_values(series: pd.Series, statistics: dict[str, float | int]) -> None:
    """Проверить воспроизводимость чисел, приведенных в отчете."""
    expected_first_15 = [
        -6352, -9181, -988, -1976, -2686,
        -5428, -6642, 7870, -7152, 9349,
        3825, -8959, -9024, -6930, -2836,
    ]
    expected_statistics = {
        "Минимальное значение": -9987,
        "Максимальное значение": 9999,
        "Сумма чисел": 172397,
        "Количество повторяющихся элементов": 23,
        "Среднеквадратическое отклонение": 5754.15,
    }

    if series.head(15).tolist() != expected_first_15:
        raise RuntimeError("Первые 15 значений не совпадают с таблицей 2.1 отчета.")

    for key, expected in expected_statistics.items():
        if statistics[key] != expected:
            raise RuntimeError(
                f"Показатель '{key}' не совпадает с отчетом: "
                f"получено {statistics[key]}, ожидалось {expected}."
            )


def run(show_plots: bool = True) -> None:
    print("=== SimpleAnalysis ===")

    # 2. Получение Dataset и создание Series.
    series = generate_dataset()
    print_dataset_info(series)
    series.to_csv(DATASET_FILE, index=False, encoding="utf-8-sig")

    # 5. Очистка данных от цифрового мусора.
    series = clean_data(series)
    print("\nПосле очистки элементов:", len(series))

    # 6–7. Количественный анализ.
    statistics = calculate_statistics(series)
    validate_report_values(series, statistics)

    # 3–4, 8. Формирование DataFrame, сводная информация, разделение по признакам.
    dataframe = build_dataframe(series)
    dataframe.to_csv(DATAFRAME_FILE, index=False, encoding="utf-8-sig")
    dataframe_summary = format_dataframe_summary(dataframe)
    classification_summary = format_classification_summary(dataframe)

    print("\nПервые строки DataFrame:")
    print(dataframe.head(10).to_string(index=False))

    # 9. Суммарные данные по диапазонам.
    ranges = build_ranges_summary(series)
    ranges.to_csv(RANGES_FILE, index=False, encoding="utf-8-sig")

    # 13. Объединённая структура показателей.
    combined = build_combined_dataframe(series)
    combined.to_csv(COMBINED_FILE, index=False, encoding="utf-8-sig")

    save_text_report(statistics, dataframe_summary, classification_summary, ranges)

    # 10–13. Визуализация: исходные, промежуточный анализ, объединённые данные.
    plot_linear(series, show_plots)
    plot_histogram(series, show_plots)
    plot_sorted(dataframe, show_plots)
    plot_combined_dependence(combined, show_plots)

    print("\nФайлы сформированы:")
    for path in [
        DATASET_FILE,
        DATAFRAME_FILE,
        RANGES_FILE,
        COMBINED_FILE,
        RESULT_FILE,
        LINEAR_PLOT_FILE,
        HISTOGRAM_FILE,
        SORTED_PLOT_FILE,
        COMBINED_PLOT_FILE,
    ]:
        print("-", path.name)

    print("\nПрограмма завершила работу корректно.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimpleAnalysis — анализ числового Dataset")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="сохранить графики в PNG без открытия окон Matplotlib",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(show_plots=not args.no_show)
