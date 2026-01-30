import { useLocalSearchParams, useRouter } from "expo-router";
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";
import { Ionicons } from "@expo/vector-icons";

export default function Confirmation() {
  const router = useRouter();
  const { success, message } = useLocalSearchParams();
  const isSuccess = success === "true";

  return (
    <View style={styles.container}>
      <View style={styles.card}>

        {/* Title AT THE TOP */}
        <Text style={[styles.title, isSuccess ? styles.successText : styles.errorText]}>
          {isSuccess ? "Appointment Confirmed!" : "Booking Failed!"}
        </Text>

        {/* Big Icon BELOW TITLE */}
        <View
          style={[
            styles.iconContainer,
            isSuccess ? styles.successBg : styles.errorBg,
          ]}
        >
          <Ionicons
            name={isSuccess ? "checkmark-circle" : "close-circle"}
            size={120}
            color={isSuccess ? "#50C878" : "#FF6B6B"}
          />
        </View>

        {/* Subtext BELOW ICON */}
        <Text style={styles.message}>
          {message ||
            (isSuccess
              ? "Your appointment has been successfully submitted. You’ll be notified once it’s confirmed."
              : "The selected timeslot is unavailable. Please try again with a different time.")}
        </Text>

        {/* Button at the bottom */}
        <TouchableOpacity
          style={[
            styles.button,
            isSuccess ? styles.successBtn : styles.errorBtn,
          ]}
          onPress={() => router.replace("/dashboard")}
        >
          <Text style={styles.buttonText}>
            {isSuccess ? "Go to Dashboard" : "Back to Dashboard"}
          </Text>
        </TouchableOpacity>

      </View>
    </View>
  );
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#E6F5FF",
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },

  card: {
    backgroundColor: "#fff",
    borderRadius: 24,
    paddingVertical: 50,
    paddingHorizontal: 30,
    alignItems: "center",     // CENTER EVERYTHING
    justifyContent: "center", // CENTER EVERYTHING
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 4 },
    elevation: 6,
    width: "90%",
    maxWidth: 380,
  },

  // BIG CENTER ICON
  iconContainer: {
  width: 200,
  height: 200,
  borderRadius: 100,
  justifyContent: "center",
  alignItems: "center",
  marginBottom: 25,
  marginTop: 10,      // new spacing
},

  successBg: {
    backgroundColor: "#E8F9EE", // light green circle bg
  },
  errorBg: {
    backgroundColor: "#FDECEC", // light red circle bg
  },

  title: {
  fontSize: 26,
  fontWeight: "900",
  textAlign: "center",
  marginBottom: 20,   // updated
  fontFamily: "LeagueSpartan",
},

  successText: {
    color: "#77CDE0",
  },
  errorText: {
    color: "##FF6B6B",
  },

  message: {
    fontSize: 16,
    textAlign: "center",
    color: "#4A4A4A",
    lineHeight: 22,
    marginBottom: 35,  // more breathing room before button
    fontFamily: "LeagueSpartan",
    paddingHorizontal: 10,
  },

  button: {
    paddingVertical: 14,
    borderRadius: 30,
    width: 220,         // slightly wider for balance
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#000",
    shadowOpacity: 0.15,
    shadowOffset: { width: 0, height: 3 },
    shadowRadius: 4,
    elevation: 4,
  },

  successBtn: {
    backgroundColor: "#77CDE0",
  },
  errorBtn: {
    backgroundColor: "#FBC02D",
  },

  buttonText: {
    color: "#fff",
    fontWeight: "bold",
    fontSize: 16,
    fontFamily: "LeagueSpartan",
  },
});
