import React from "react";
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from "react-native";
import { MaterialIcons, Feather, Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";


export default function EyePhotoInstructions() {
  const router = useRouter();

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <MaterialIcons name="arrow-back-ios" size={24} color="#007EF2" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Eye Photo Instructions</Text>
      </View>

      {/* Title */}
      <Text style={styles.title}>How to Take an Eye Photo</Text>

      {/* Step 1 */}
      <View style={styles.card}>
        <Feather name="sun" size={28} color="#007EF2" />
        <Text style={styles.cardTitle}>1. Find Good Lighting</Text>
        <Text style={styles.cardText}>
          Stand in a bright area. Face a window or a light source. Avoid shadows or dark places.
        </Text>
      </View>

      {/* Step 2 */}
      <View style={styles.card}>
        <MaterialIcons name="stay-current-portrait" size={30} color="#007EF2" />
        <Text style={styles.cardTitle}>2. Position the Camera</Text>
        <Text style={styles.cardText}>
          Hold your phone 10–15 cm from your eye. Keep it directly in front so the whole eye fits in the frame.
        </Text>
      </View>

      {/* Step 3 */}
      <View style={styles.card}>
        <Ionicons name="person-outline" size={30} color="#007EF2" />
        <Text style={styles.cardTitle}>3. Keep Your Head Still</Text>
        <Text style={styles.cardText}>Look straight at the camera. Avoid shaking or blinking.</Text>
      </View>

      {/* Step 4 */}
      <View style={styles.card}>
        <MaterialIcons name="visibility" size={30} color="#007EF2" />
        <Text style={styles.cardTitle}>4. Open Your Eyes Wide</Text>
        <Text style={styles.cardText}>Keep your eyes wide open so the iris and pupil are clear.</Text>
      </View>

      {/* Step 5 */}
      <View style={styles.card}>
        <Feather name="eye-off" size={28} color="#007EF2" />
        <Text style={styles.cardTitle}>5. Remove Anything Blocking Your Eyes</Text>
        <Text style={styles.cardText}>Remove glasses and push your hair aside. Avoid reflections.</Text>
      </View>

      {/* Step 6 */}
      <View style={styles.card}>
        <MaterialIcons name="center-focus-weak" size={30} color="#007EF2" />
        <Text style={styles.cardTitle}>6. Make the Image Clear</Text>
        <Text style={styles.cardText}>Tap on your eye to focus. Make sure the picture is bright and not blurry.</Text>
      </View>

      {/* Step 7 */}
      <View style={styles.card}>
        <MaterialIcons name="camera-enhance" size={30} color="#007EF2" />
        <Text style={styles.cardTitle}>7. Capture These Angles</Text>
        <Text style={styles.cardText}>• Front view (required)</Text>
        <Text style={styles.cardText}>• Left angle</Text>
        <Text style={styles.cardText}>• Right angle</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F0F4F8",
  },
  content: {
    padding: 20,
    paddingBottom: 60,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 15,
  },
  backButton: {
    paddingRight: 10,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: "#007EF2",
  },
  title: {
    fontSize: 24,
    fontWeight: "700",
    color: "#007EF2",
    marginBottom: 20,
    textAlign: "center",
  },
  card: {
    backgroundColor: "#fff",
    padding: 20,
    borderRadius: 20,
    marginBottom: 18,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 8,
    elevation: 4,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: "#007EF2",
    marginTop: 10,
  },
  cardText: {
    fontSize: 15,
    color: "#333",
    marginTop: 4,
  },
});
