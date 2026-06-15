'''AGENT'''

class LiveAuctionAgent:
    def __init__(self):
        self.bankroll = 500000.0
        self.predicted_value = 0.0

        self.current_round = 1

        base_path = os.path.dirname(os.path.abspath(__file__))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(os.path.join(base_path, "model_TakuTajoJuly.pkl"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(base_path, "encoders_TakuTajoJuly.pkl"), "rb") as f:
                self.encoders = pickle.load(f)


    def analyze_item(self, item_features: dict):
        feature_cols = ['year', 'make', 'model', 'trim', 'body', 'transmission', 
                        'condition', 'odometer', 'color', 'interior']
        features = []
        for col in feature_cols:
            val = item_features.get(col, None)

            if col in ['year', 'condition', 'odometer']:
                #for numeric columns
                features.append(float(val) if val is not None else 0.0)
            else:
                # Categorical columns (Applying our Target Encoding)
                target_mapper = self.encoders[col]

                if val in target_mapper:
                    features.append(float(target_mapper[val]))
                else:
                    features.append(float(target_mapper.median()))

        input_data = np.array([features])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_prediction = self.model.predict(input_data)[0]
            self.predicted_value = float(np.expm1(log_prediction))

        self.current_round = 1


    def place_bid(self, current_highest_bid: float) -> float:
        """Deterministic Bid Sequence: The Asymptotic Gap Strategy"""
        target_margin = 0.85 
        max_willing_to_pay = min(self.predicted_value * target_margin, self.bankroll) #we have a 15% profit margin

        gap = max_willing_to_pay - current_highest_bid
        if gap <= 0:
            return 0.0  # Fold. we drop out if the price exceeds our profit margin or bankroll

        round_weight = min(0.5, self.current_round * 0.10)

        bid_increment = gap * round_weight
        next_bid = current_highest_bid + bid_increment

        self.current_round += 1 #Increment the round tracker in case someone outbids us and we have to bid again

        return round(next_bid, 2)


    def auction_result(self, won: bool, winning_bid: float, actual_price: float, current_bankroll: float):
        """Updates the bankroll at the end of the car's auction."""
        self.bankroll = current_bankroll
        self.current_round = 1        
