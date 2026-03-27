import React from "react";
import { View, Text, Image, StyleSheet, TouchableOpacity, ScrollView, Dimensions } from "react-native";
import { useRouter, useLocalSearchParams } from "expo-router";
import { MaterialIcons, Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';

/// =======================================================
// 1. DEFINE ALL CONSTANTS (Colors & Styles)
// =======================================================

// Brand Colors
const BRAND_BLUE = "#0057B7";

// Status Colors
const ATTENTION_COLOR = "#FF6B6B";   // For warnings or failed status
const NORMAL_COLOR = "#50C878";      // For normal/healthy status

// UI Accent Colors
const PRIMARY_ACTION_COLOR = "#FFD54F"; // Yellow buttons or highlights
const BACKGROUND_COLOR = "#E8F7FF";     // Light background
const NOTE_BG_COLOR = "#D9ECFF";        // Info/notes background
const INFO_BORDER_COLOR = "#77CDE0";    // Light blue border/accent color (NEW)

// Neutral Colors
const NEUTRAL_TEXT = "#333333";
const SECONDARY_BORDER = "#99AACC";
const CARD_BG = "#FFFFFF";


export default function EyeResults() {
  const router = useRouter();
  const params = useLocalSearchParams();

  // Destructure
  const { status, message, diagnosis, imageUri, timestamp } = params;

  const isAttention = status === "requires_attention";
  const bannerColor = isAttention ? ATTENTION_COLOR : NORMAL_COLOR;

  const navigateToAppointment = () => {
    router.push({
      pathname: "/book-appointment",
      params: {
        reason: `AI Screening: ${diagnosis}`,
        aiScreening: true,
        imageUri,
      },
    });
  };

  const navigateToDashboard = () => router.push("/dashboard");
  const handleBack = () => router.back();

  const formattedTimestamp = timestamp
    ? new Date(timestamp).toLocaleDateString(undefined, {
        year: 'numeric', month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit',
      })
    : "Just now";

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: BACKGROUND_COLOR }}>
      {/* FIXED HEADER VIEW */}
      <View style={styles.fixedHeader}>
        <TouchableOpacity style={styles.backButton} onPress={handleBack}>
          <Ionicons name="arrow-back" size={24} color={BRAND_BLUE} />
        </TouchableOpacity>
        <Text style={styles.pageTitleFixed}>Eye Results</Text>
      </View>

      {/* SCROLLABLE CONTENT */}
      <ScrollView contentContainerStyle={styles.container}>

        {/* RESULT BANNER */}
        <View style={[styles.banner, { backgroundColor: bannerColor }]}>
          <MaterialIcons
            name={isAttention ? "error-outline" : "check-circle-outline"}
            size={70}
            color="#fff"
          />
          <View style={styles.bannerTextWrapper}>
            <Text style={styles.bannerTitle}>
              {isAttention ? "ACTION REQUIRED" : "SCREENING COMPLETE"}
            </Text>
            <Text style={styles.bannerSubtitle}>
              {diagnosis}
            </Text>
          </View>
        </View>

        {/* SCREENING STATUS CARD */}
        <View style={styles.card}>
          <View style={styles.diagnosisHeader}>
            <Text style={styles.diagnosisLabel}>Screening Status</Text>
            <Text style={[styles.diagnosisStatus, {color: bannerColor}]}>
              {isAttention ? "Consultation Recommended" : "No Issues Detected"}
            </Text>
          </View>
          <View style={styles.separator} />
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Date & Time</Text>
            <Text style={styles.detailValue}>{formattedTimestamp}</Text>
          </View>
        </View>

        {/* IMAGE CARD - UPDATED FOR NO ZOOM */}
        <View style={styles.imageCard}>
          <Text style={styles.imageCardTitle}>Analyzed Image</Text>
          {imageUri && (
            <View style={styles.imageContainer}>
                <Image
                source={{ uri: decodeURIComponent(imageUri) }}
                style={styles.image}
                // [UPDATED] 'contain' ensures the WHOLE image fits without zooming/cropping
                resizeMode="contain" 
                />
            </View>
          )}
        </View>

        {/* MESSAGE CARD */}
        <View style={styles.card}>
          <Text style={styles.messageBodyTitle}>AI Preliminary Report</Text>
          <Text style={styles.messageBody}>{message}</Text>
        </View>

        {/* NOTE WARNING */}
        <View style={[styles.noteContainer, { backgroundColor: NOTE_BG_COLOR }]}>
          <Text style={[styles.noteText, { color: BRAND_BLUE }]}>
            <Text>ℹ️ This is an </Text>
            <Text style={{ fontWeight: 'bold' }}>AI-assisted preliminary screening</Text>
            <Text> only and is </Text>
            <Text style={{ fontWeight: 'bold' }}>not a medical diagnosis</Text>
            <Text>. Always consult a qualified physician for professional medical advice.</Text>
          </Text>
        </View>

        {/* ACTION BUTTONS */}
        <View style={styles.actionContainer}>
          {isAttention ? (
            <TouchableOpacity style={[styles.buttonPrimary, { backgroundColor: PRIMARY_ACTION_COLOR }]} onPress={navigateToAppointment}>
              <Text style={[styles.buttonPrimaryText, { color: NEUTRAL_TEXT }]}>BOOK APPOINTMENT NOW</Text>
              <MaterialIcons name="chevron-right" size={24} color={NEUTRAL_TEXT} />
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={[styles.buttonSecondary, { borderColor: SECONDARY_BORDER, backgroundColor: '#FFF' }]} onPress={navigateToDashboard}>
              <Text style={[styles.buttonSecondaryText, { color: BRAND_BLUE }]}>BACK TO HOME</Text>
            </TouchableOpacity>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

// =======================================================
// 🎨 STYLES
// =======================================================
const styles = StyleSheet.create({
  container: {
    padding: 20,
    paddingTop: 10,
    paddingBottom: 40,
  },
  fixedHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: BACKGROUND_COLOR,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  backButton: {
    padding: 5,
    marginRight: 15,
    marginTop: 25,
  },
  pageTitleFixed: {
    fontSize: 20,
    fontWeight: 'bold',
    color: BRAND_BLUE,
    marginTop: 25,
  },
  
  // BANNER
  banner: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    borderRadius: 15,
    marginBottom: 20,
    elevation: 4,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  bannerTextWrapper: {
    marginLeft: 15,
    flex: 1,
  },
  bannerTitle: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    opacity: 0.9,
    marginBottom: 4,
  },
  bannerSubtitle: {
    color: '#fff',
    fontSize: 22,
    fontWeight: 'bold',
  },

  // CARDS
  card: {
    backgroundColor: BACKGROUND_COLOR ,
    borderRadius: 12,
    padding: 7,
    marginBottom: 20,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    marginTop: -15,

  },
  imageCard: {
    backgroundColor: NOTE_BG_COLOR,
    borderRadius: 12,
    padding: 15,
    marginBottom: 20,
    elevation: 2,
    alignItems: 'center',
    marginTop: -15,
  },
  imageCardTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: NEUTRAL_TEXT,
    marginBottom: 10,
    alignSelf: 'flex-start',
  },
  imageContainer: {
    width: '100%',
    height: 120,
    backgroundColor: '#F0F0F0', // Light grey background to show boundaries
    borderRadius: 8,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },

  // DETAILS
  diagnosisHeader: {
    marginBottom: 10,
  },
  diagnosisLabel: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  diagnosisStatus: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  separator: {
    height: 1,
    backgroundColor: '#EEE',
    marginVertical: 12,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  detailLabel: {
    fontSize: 14,
    color: '#666',
  },
  detailValue: {
    fontSize: 14,
    fontWeight: '600',
    color: NEUTRAL_TEXT,
  },

  // MESSAGE
  messageBodyTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: NEUTRAL_TEXT,
    marginBottom: 8,
  },
  messageBody: {
    fontSize: 15,
    color: '#444',
    lineHeight: 22,
  },

  // NOTE
  noteContainer: {
    padding: 15,
    borderRadius: 10,
    marginBottom: 25,
    marginTop: -15,
  },
  noteText: {
    fontSize: 13,
    textAlign: 'center',
    lineHeight: 18,
  },

  // ACTIONS
  actionContainer: {
    marginTop: 10,
  },
  buttonPrimary: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 12,
    borderRadius: 12,
    elevation: 3,
    marginTop: -25,
  },
  buttonPrimaryText: {
    fontSize: 16,
    fontWeight: 'bold',
    marginRight: 5,
  },
  buttonSecondary: {
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 16,
    borderRadius: 12,
    borderWidth: 2,
  },
  buttonSecondaryText: {
    fontSize: 16,
    fontWeight: 'bold',
  },
});
