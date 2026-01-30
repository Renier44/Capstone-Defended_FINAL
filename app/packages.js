// app/packages.js
import React from "react";
import { View, Text, StyleSheet, ScrollView, Image } from "react-native";

export default function PackagesScreen() {
  const packages = [
    {
      name: "BASIC",
      price: "P500",
      services: [
        "REFRACTION",
        "INTERNAL EYE SCREEN",
        "EXTERNAL EYE SCREEN",
        "MEDICAL CERTIFICATE",
      ],
    },
    {
      name: "PREMIUM",
      price: "P750",
      services: [
        "REFRACTION",
        "INTERNAL EYE SCREEN",
        "EXTERNAL EYE SCREEN",
        "EYE PRESSURE TEST (GLAUCOMA SCREENING)",
        "MEDICAL CERTIFICATE",
      ],
    },
    {
      name: "STUDENT/WORK",
      price: "P650",
      services: [
        "REFRACTION",
        "INTERNAL EYE SCREEN",
        "EXTERNAL EYE SCREEN",
        "COLOR VISION TESTING",
        "MEDICAL CERTIFICATE",
      ],
    },
    {
      name: "DRY EYE TEST",
      price: "P550",
      services: [
        "SLIT LAMP",
        "INTERNAL EYE SCREEN",
        "EXTERNAL EYE SCREEN",
        "SCHIRMER TEST",
        "FLUORESCEIN STAINING",
        "MEIBOGRAPHY",
      ],
    },
    {
      name: "CHILDREN",
      price: "P1,250",
      subtitle: "by appointment basis only",
      services: [
        "REFRACTION",
        "INTERNAL EYE SCREEN",
        "EXTERNAL EYE SCREEN",
        "COLOR VISION TESTING",
        "MYOPIA MANAGEMENT",
        "STRABISMUS/LAZY EYE SCREENING",
        "MEDICAL CERTIFICATE",
      ],
    },
  ];

  return (
    <View style={styles.container}>
      {/* Top Header */}
      <Text style={styles.headerText}>PACKAGES</Text>

      {/* Main Content */}
      <ScrollView contentContainerStyle={styles.content}>
        {packages.map((pkg, index) => (
          <View key={index} style={styles.packageBox}>
            <View style={styles.packageHeader}>
              <Text style={styles.packageName}>{pkg.name}</Text>
              <Text style={styles.packagePrice}>{pkg.price}</Text>
            </View>
            {pkg.subtitle && (
              <Text style={styles.packageSubtitle}>{pkg.subtitle}</Text>
            )}
            {pkg.services.map((service, i) => (
              <Text key={i} style={styles.packageService}>
                {service}
              </Text>
            ))}
          </View>
        ))}

        {/* Faint Background Logo */}
        <Image
          source={require("../assets/images/icon.png")}
          style={styles.backgroundLogo}
        />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#8ED7F0", // Soft blue background
  },
  headerText: {
    fontSize: 28,
    fontWeight: "bold",
    color: "#fff",
    textAlign: "center",
    marginVertical: 20,
    letterSpacing: 2,
    fontFamily: "Poppins-Bold", // Works if you load Poppins font, otherwise fallback
  },
  content: {
    flexGrow: 1,
    padding: 20,
    paddingBottom: 100,
  },
  packageBox: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 15,
    marginBottom: 20,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
    elevation: 3, // For Android shadow
  },
  packageHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
  },
  packageName: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#65A3D5", // Light blue
  },
  packagePrice: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#FFD700", // Golden yellow
  },
  packageSubtitle: {
    fontSize: 14,
    fontStyle: "italic",
    color: "#777",
    marginBottom: 5,
  },
  packageService: {
    fontSize: 14,
    color: "#444", // Darker gray for readability
    marginVertical: 2,
  },
  backgroundLogo: {
    position: "absolute",
    right: 30,
    bottom: 30,
    width: 180,
    height: 180,
    opacity: 0.06, // Ghost effect
    resizeMode: "contain",
  },
});
