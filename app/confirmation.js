import { useLocalSearchParams, useRouter } from "expo-router";
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";

export default function Confirmation() {
  const router = useRouter();
  const { success, message } = useLocalSearchParams();

  const isSuccess = success === "true";

  return (
    <View style={styles.container}>
      {/* Icon */}
      <View
        style={[
          styles.iconContainer,
          isSuccess ? styles.successBg : styles.errorBg,
        ]}
      >
        <Text style={styles.icon}>{isSuccess ? "✓" : "✕"}</Text>
      </View>

      {/* Message */}
      <Text style={styles.message}>
        {message ||
          (isSuccess
            ? "Your appointment has been submitted successfully."
            : "The selected timeslot is no longer available. Please choose another one.")}
      </Text>

      {/* Button */}
      <TouchableOpacity
        style={[styles.button, isSuccess ? styles.successBtn : styles.errorBtn]}
        onPress={() => {
          if (isSuccess) {
            router.replace("/dashboard");
          } else {
            router.back();
          }
        }}
      >
        <Text style={styles.buttonText}>
          {isSuccess ? "Go Home" : "Try Again"}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#C8EAF7", // 🟢 same background everywhere
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  iconContainer: {
    width: 120,
    height: 120,
    borderRadius: 60,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 20,
    backgroundColor: "#fff", // keep white circle para clean contrast
  },
  icon: {
    fontSize: 60,
    fontWeight: "bold",
    color: "#2260FF", // main accent color
  },
  message: {
    fontSize: 18,
    textAlign: "center",
    marginBottom: 20,
    fontWeight: "600",
    color: "#2260FF",
    fontFamily: "LeagueSpartan",
  },
  button: {
    padding: 14,
    borderRadius: 30,
    width: 160,
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.15,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 4,
    elevation: 3,
  },
  successBtn: {
    backgroundColor: "#4CAF50",
  },
  errorBtn: {
    backgroundColor: "#FFD700",
  },
  buttonText: {
    color: "#fff",
    fontWeight: "bold",
    fontSize: 16,
    fontFamily: "LeagueSpartan",
  },
});
