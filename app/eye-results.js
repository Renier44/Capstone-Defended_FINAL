// app/eye-result.js
import React from "react";
import { View, Text, Image, TouchableOpacity, StyleSheet } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";

export default function EyeResultScreen() {
  const router = useRouter();
  const { status, message, imageUri, timestamp } = useLocalSearchParams();

  const isAttention = status === "requires_attention";

  return (
    <View style={styles.container}>
      {/* Banner */}
      <View
        style={[
          styles.banner,
          { backgroundColor: isAttention ? "#d9534f" : "#5cb85c" },
        ]}
      >
        <Text style={styles.bannerText}>Your Eyes Status</Text>
        <Text style={styles.bannerSub}>
          {isAttention ? "Requires Attention" : "Normal"}
        </Text>
      </View>

      {/* Card */}
      <View style={styles.card}>
        {imageUri ? (
          <Image source={{ uri: imageUri }} style={styles.image} />
        ) : (
          <Text style={{ color: "#999", marginBottom: 10 }}>
            No image provided
          </Text>
        )}

        <Text style={styles.date}>
          {timestamp || new Date().toLocaleString()}
        </Text>
        <Text style={styles.title}>Initial Result</Text>
        <Text style={styles.message}>
          {message ||
            (isAttention
              ? "Potential indicator of a binocular vision problem detected. Further examination recommended."
              : "Normal")}
        </Text>
      </View>

      {/* Action button */}
      <TouchableOpacity
        style={styles.button}
        onPress={() =>
          isAttention
            ? router.push({
                pathname: "/book-appointment",
                params: {
                  reason: message,       // Pass AI initial result
                  aiScreening: true,     // Flag to detect AI flow
                  imageUri: imageUri,    // Optional: send image for reference
                  timestamp: timestamp,  // Optional: send timestamp
                },
              })
            : router.push("/dashboard")
        }
      >
        <Text style={styles.buttonText}>
          {isAttention ? "BOOK AN APPOINTMENT" : "BACK TO DASHBOARD"}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, backgroundColor: "#fff" },
  banner: {
    padding: 16,
    borderRadius: 8,
    alignItems: "center",
  },
  bannerText: { fontSize: 18, color: "#fff" },
  bannerSub: { fontSize: 22, color: "#fff", fontWeight: "bold" },
  card: {
    marginTop: 20,
    backgroundColor: "#f9f9f9",
    padding: 16,
    borderRadius: 8,
    alignItems: "center",
  },
  image: { width: 200, height: 120, borderRadius: 8, marginBottom: 10 },
  date: { color: "#888" },
  title: { fontSize: 16, fontWeight: "bold", marginTop: 8 },
  message: { marginTop: 4, textAlign: "center" },
  button: {
    marginTop: 30,
    backgroundColor: "gold",
    padding: 14,
    borderRadius: 8,
  },
  buttonText: { textAlign: "center", fontWeight: "600" },
});
