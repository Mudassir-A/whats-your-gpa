# What's Your GPA?

This project calculates and predicts GPA based on various input parameters using machine learning techniques.

## Directory Structure

```
whats-your-GPA/
├── data/
│   ├── ...
├── models/
├── src/
│   ├── model.py
│   ├── train.py
├── requirements.txt
├── README.md
```

- `data/`: Contains the data.
- `models/`: Stores trained models.
- `src/`: Source code for data preprocessing, model training, and GPA prediction.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: Project documentation.

## Usage Steps

1. Clone the repository:
	```bash
	git clone https://github.com/Mudassir-A/whats-your-gpa.git
	cd whats-your-gpa
	```

2. Create a virtual environment and activate it:
	```bash
	python -m venv venv
	source venv/bin/activate  # On Windows use `venv\Scripts\activate`
	```

3. Install the required packages:
	```bash
	pip install -r requirements.txt
	```

4. Predict GPA:
	```bash
	python src/train.py
	```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.