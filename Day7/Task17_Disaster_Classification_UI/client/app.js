const API_BASE_URL = "http://127.0.0.1:5000";


function onClickedClassifyMessage() {
  console.log("Analyze Transmission button clicked");
  
  const textVal = document.getElementById("uiMessage").value;
  if (!textVal || textVal.trim() === "") {
    alert("Please enter or paste an emergency transmission message first.");
    return;
  }
  
  const btn = document.getElementById("btnClassify");
  const spinnerIcon = btn.querySelector(".spinner-icon");
  const btnText = btn.querySelector("span span");
  
  btn.disabled = true;
  spinnerIcon.className = "fa-solid fa-compact-disc spinner-icon spinning";
  btnText.innerText = "SCANNING SIGNAL...";
  
  const url = `${API_BASE_URL}/predict_disaster`;
  
  $.post(url, {
      text: textVal
  }, function(data, status) {
      console.log("Prediction data:", data);
      
      btn.disabled = false;
      spinnerIcon.className = "fa-solid fa-microchip-ai spinner-icon";
      btnText.innerText = "ANALYZE TRANSMISSION";
      
      if(data) {
          showClassificationResult(data);
      }
  }).fail(function() {
      alert("Error contacting the prediction server. Please verify the Flask server is running.");
      btn.disabled = false;
      spinnerIcon.className = "fa-solid fa-microchip-ai spinner-icon";
      btnText.innerText = "ANALYZE TRANSMISSION";
  });
}

function showClassificationResult(data) {
  const placeholder = document.getElementById("resultPlaceholder");
  const resultContent = document.getElementById("resultContent");
  const resultCard = document.getElementById("resultCard");
  
  placeholder.classList.add("hidden");
  resultContent.classList.remove("hidden");
  resultCard.classList.add("active");
  
  const statusElem = document.getElementById("uiDisasterStatus");
  statusElem.className = "metric-value badge"; // reset classes
  
  if (data.is_disaster) {
      statusElem.innerText = "🚨 DISASTER DETECTED";
      statusElem.classList.add("status-disaster");
  } else {
      statusElem.innerText = "🟢 NOMINAL - SAFE";
      statusElem.classList.add("status-non-disaster");
  }
  
  const binConfidence = data.binary_confidence;
  document.getElementById("uiBinaryConfidenceText").innerText = `${binConfidence.toFixed(2)}%`;
  document.getElementById("uiBinaryConfidenceFill").style.width = `${binConfidence}%`;
  
  const categoryBypassed = document.getElementById("categoryBypassedView");
  const categoryActive = document.getElementById("categoryActiveView");
  
  if (!data.is_disaster) {
    
      categoryActive.classList.add("hidden");
      categoryBypassed.classList.remove("hidden");
  } else {
      
      categoryBypassed.classList.add("hidden");
      categoryActive.classList.remove("hidden");
      
      const category = data.category;
      const confidence = data.category_confidence;
      const categoryElem = document.getElementById("uiCategory");
      
      const formattedCategory = category.replace(/_/g, " ").toUpperCase();
      categoryElem.innerText = formattedCategory;
      
      categoryElem.className = "metric-value badge"; 
      
      const alertTitle = document.getElementById("uiAlertTitle");
      const alertMsg = document.getElementById("uiAlertMessage");
      const alertContainer = document.getElementById("actionAlert");
      
      const catLower = category.toLowerCase();
      
      if (catLower === "injured_or_dead_people") {
          categoryElem.classList.add("threat-danger");
          alertTitle.innerText = "CRITICAL MEDICAL EMERGENCY";
          alertMsg.innerText = "Dispatching Emergency Medical Services (EMS) and Search & Rescue immediately.";
          alertContainer.style.background = "rgba(255, 51, 102, 0.05)";
          alertContainer.style.borderColor = "rgba(255, 51, 102, 0.2)";
      } else if (catLower === "infrastructure_and_utilities_damage") {
          categoryElem.classList.add("threat-danger");
          alertTitle.innerText = "INFRASTRUCTURE EMERGENCY";
          alertMsg.innerText = "Alerting public works and utility dispatchers. Blocked roads or grid failures detected.";
          alertContainer.style.background = "rgba(255, 51, 102, 0.05)";
          alertContainer.style.borderColor = "rgba(255, 51, 102, 0.2)";
      } else if (catLower === "requests_or_needs") {
          categoryElem.classList.add("threat-cyan");
          alertTitle.innerText = "RESOURCE REQUISITION ALERT";
          alertMsg.innerText = "Supplies needed (food, water, medicine, or shelter). Coordinating relief delivery.";
          alertContainer.style.background = "rgba(0, 242, 254, 0.05)";
          alertContainer.style.borderColor = "rgba(0, 242, 254, 0.2)";
      } else if (catLower === "affected_individual") {
          categoryElem.classList.add("threat-warning");
          alertTitle.innerText = "AFFECTED PERSON REPORTED";
          alertMsg.innerText = "Stranded or missing individuals flagged. Deploying local monitoring and rescue teams.";
          alertContainer.style.background = "rgba(255, 153, 0, 0.05)";
          alertContainer.style.borderColor = "rgba(255, 153, 0, 0.2)";
      } else if (catLower === "donation_and_volunteering") {
          categoryElem.classList.add("threat-success");
          alertTitle.innerText = "MUTUAL AID TRANSMISSION";
          alertMsg.innerText = "Volunteer assistance or donation supplies offered. Routing details to dispatch base.";
          alertContainer.style.background = "rgba(0, 255, 204, 0.05)";
          alertContainer.style.borderColor = "rgba(0, 255, 204, 0.2)";
      } else {
          categoryElem.classList.add("threat-warning");
          alertTitle.innerText = "UNCLASSIFIED DISPATCH ALERT";
          alertMsg.innerText = "Verifying report details. Flagged for secondary human supervisor evaluation.";
          alertContainer.style.background = "rgba(255, 153, 0, 0.05)";
          alertContainer.style.borderColor = "rgba(255, 153, 0, 0.2)";
      }
    
      document.getElementById("uiConfidenceText").innerText = `${confidence.toFixed(2)}%`;
      document.getElementById("uiConfidenceFill").style.width = `${confidence}%`;
  }
}

function onPageLoad() {
  console.log("document loaded");

  
  $(".btn-sample").click(function() {
    const textVal = $(this).attr("data-text");
    $("#uiMessage").val(textVal);
    $("#uiMessage").focus();
  });
  const url = `${API_BASE_URL}/get_categories`;
  $.get(url, function(data, status) {
    if(data && data.categories) {
      const categories = data.categories;
      const listContainer = $("#uiCategoriesList");
      listContainer.empty();
      
      categories.forEach(function(cat) {
        const formatted = cat.replace(/_/g, " ");
        const badge = $(`<span class="category-badge">${formatted}</span>`);
        listContainer.append(badge);
      });
    }
  }).fail(function() {
      console.warn("Could not retrieve categories. Server offline.");
      $("#uiCategoriesList").html('<span style="color:var(--color-danger);font-size:0.9rem;"><i class="fa-solid fa-triangle-exclamation"></i> Server offline. Start server.py.</span>');
  });
}

window.onload = onPageLoad;
